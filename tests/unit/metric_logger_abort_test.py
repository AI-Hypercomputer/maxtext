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

"""Tests for monitoring metrics"""
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

from maxtext.common.metric_logger import MetricLogger


class MetricLoggerAbortTest(unittest.TestCase):
  def _make_logger(self, abort_on_nan_loss, abort_on_inf_loss):
    logger = MetricLogger.__new__(MetricLogger)  # skip __init__
    logger.config = SimpleNamespace(
        abort_on_nan_loss=abort_on_nan_loss,
        abort_on_inf_loss=abort_on_inf_loss,
        enable_tensorboard=True,
        metrics_file="/tmp/fake_metrics.jsonl",
        gcs_metrics=True,
        managed_mldiagnostics=True,
    )
    return logger

  def _metrics(self, loss):
    return {"scalar": {"learning/loss": loss}}

  @mock.patch("jax.process_index", return_value=0)
  def test_abort_on_nan_exits_after_writes(self, _):
    logger = self._make_logger(True, False)

    with (
        mock.patch.object(logger, "log_metrics") as log_metrics,
        mock.patch.object(logger, "write_metrics_to_tensorboard") as tb,
        mock.patch.object(logger, "write_metrics_locally") as local,
        mock.patch.object(logger, "write_metrics_for_gcs") as gcs,
        mock.patch.object(logger, "write_metrics_to_managed_mldiagnostics") as mldiag,
    ):
      with self.assertRaises(SystemExit) as cm:
        logger.write_metrics(self._metrics(np.nan), step=1, is_training=True)

    self.assertEqual(cm.exception.code, 1)
    log_metrics.assert_called_once()
    tb.assert_called_once()
    local.assert_called_once()
    gcs.assert_called_once()
    mldiag.assert_called_once()

  @mock.patch("jax.process_index", return_value=0)
  def test_abort_on_inf_exits_after_writes(self, _):
    logger = self._make_logger(False, True)
    with mock.patch.object(logger, "log_metrics"), \
         mock.patch.object(logger, "write_metrics_to_tensorboard"), \
         mock.patch.object(logger, "write_metrics_locally"), \
         mock.patch.object(logger, "write_metrics_for_gcs"), \
         mock.patch.object(logger, "write_metrics_to_managed_mldiagnostics"):
      with self.assertRaises(SystemExit):
        logger.write_metrics(self._metrics(np.inf), step=1, is_training=True)

  def test_finite_loss_does_not_exit(self):
    logger = self._make_logger(True, True)
    with mock.patch.object(logger, "log_metrics"), \
         mock.patch.object(logger, "write_metrics_to_tensorboard"), \
         mock.patch.object(logger, "write_metrics_locally"), \
         mock.patch.object(logger, "write_metrics_to_managed_mldiagnostics"), \
         mock.patch("jax.process_index", return_value=1):  # skip gcs branch
      logger.write_metrics(self._metrics(1.23), step=1, is_training=True)

  def test_abort_flags_disabled_does_not_exit(self):
    logger = self._make_logger(False, False)
    with mock.patch.object(logger, "log_metrics"), \
         mock.patch.object(logger, "write_metrics_to_tensorboard"), \
         mock.patch.object(logger, "write_metrics_locally"), \
         mock.patch.object(logger, "write_metrics_to_managed_mldiagnostics"), \
         mock.patch("jax.process_index", return_value=1):
      logger.write_metrics(self._metrics(np.nan), step=1, is_training=True)
