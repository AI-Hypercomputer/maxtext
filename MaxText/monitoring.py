"""Utilities to monitor and upload goodput data to TensorBoard asynchronously."""

import datetime
import logging
import threading
import time

import jax
from ml_goodput_measurement import goodput
from tensorboardX import writer


_exit_flag = threading.Event()
_exit_flag.clear()
_goodput_upload_thread = None
logger = logging.getLogger(__name__)


class GoodputMonitor:
  """Queries and uploads goodput data to TensorBoard at a regular interval."""

  def __init__(
      self,
      job_name: str,
      logger_name: str,
      tensorboard_dir: str,
      upload_interval: int,
  ):
    """Initializes the GoodputMonitor.

    Args:
        job_name: The name of the job to monitor.
        logger_name: The name of the Google Cloud Logging logger to use.
        tensorboard_dir: The directory to write TensorBoard data to.
        upload_interval: The interval to upload data to TensorBoard.
    """
    self._job_name = job_name
    self._logger_name = logger_name
    self._tensorboard_dir = tensorboard_dir
    self._upload_interval = upload_interval
    self._goodput_calculator = goodput.GoodputCalculator(
        job_name=self._job_name,
        logger_name=self._logger_name,
    )
    self._writer = (
        writer.SummaryWriter(self._tensorboard_dir)
        if jax.process_index() == 0
        else None
    )
    self._uploader_flag = False

  def __del__(self):
    if self._uploader_flag:
      self.stop_uploader()

  def _write_to_tensorboard(self, job_goodput: float, step):
    if self._writer is not None:
      timestamp = datetime.datetime.now().timestamp()
      self._writer.add_scalar('goodput', job_goodput, step)
      logger.info('Job goodput uploaded to Tensorboard.')
      self._writer.flush()

  def _query_and_upload_goodput(self):
    """Queries and uploads goodput data to TensorBoard."""
    time.sleep(10)
    step = 0
    while not _exit_flag.is_set():
      time.sleep(self._upload_interval)
      try:
        logger.info('Querying goodput for job: %s and logger: %s', self._job_name, self._logger_name)
        job_goodput = self._goodput_calculator.get_job_goodput()
        logger.info('Job goodput: %f', job_goodput)
        step += self._upload_interval
        self._write_to_tensorboard(job_goodput, step)
      except Exception as e:
        logger.error(
            'Error while querying and uploading goodput to Tensorboard. This'
            ' will not impact the workload.'
        )
        logging.exception(e)
        

  def start_uploader(self):
    """Starts the goodput uploader thread."""
    if self._uploader_flag:
      raise RuntimeError('Goodput uploader thread is already running.')
    self._uploader_flag = True
    _exit_flag.clear()
    global _goodput_upload_thread
    _goodput_upload_thread = threading.Thread(
        target=self._query_and_upload_goodput, daemon=True
    )
    logger.info('Starting goodput query and uploader thread in the background.')
    _goodput_upload_thread.start()

  def stop_uploader(self):
    """Stops the goodput uploader thread."""
    if not self._uploader_flag:
      raise RuntimeError('Goodput uploader thread is not running.')
    self._uploader_flag = False
    _exit_flag.set()
    if _goodput_upload_thread is not None:
      logger.info('Waiting for goodput query and uploader thread to complete.')
      _goodput_upload_thread.join()
    logger.info(
        'Goodput query and uploader thread stopped. No more goodput data will'
        ' be uploaded to Tensorboard.'
    )
