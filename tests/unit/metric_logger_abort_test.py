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
from collections import defaultdict
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

from maxtext.common.metric_logger import MetricLogger, MetadataKey

# pylint: disable=missing-function-docstring


class MetricLoggerAbortTest(unittest.TestCase):

  def _make_logger(self, abort_on_nan_loss, abort_on_inf_loss):
    """Helper to create a MetricLogger with mocked config."""
    logger = MetricLogger.__new__(MetricLogger)  # skip __init__
    logger.config = SimpleNamespace(
        abort_on_nan_loss=abort_on_nan_loss,
        abort_on_inf_loss=abort_on_inf_loss,
        enable_tensorboard=True,
        metrics_file="/tmp/fake_metrics.jsonl",
        gcs_metrics=True,
        managed_mldiagnostics=True,
        enable_wandb=False,
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
        logger.write_metrics(self._metrics(np.nan), step=1, metric_type="train")

    self.assertEqual(cm.exception.code, 1)
    log_metrics.assert_called_once()
    tb.assert_called_once()
    local.assert_called_once()
    gcs.assert_called_once()
    mldiag.assert_called_once()

  @mock.patch("jax.process_index", return_value=0)
  def test_abort_on_inf_exits_after_writes(self, _):
    logger = self._make_logger(False, True)
    with (
        mock.patch.object(logger, "log_metrics"),
        mock.patch.object(logger, "write_metrics_to_tensorboard"),
        mock.patch.object(logger, "write_metrics_locally"),
        mock.patch.object(logger, "write_metrics_for_gcs"),
        mock.patch.object(logger, "write_metrics_to_managed_mldiagnostics"),
    ):
      with self.assertRaises(SystemExit):
        logger.write_metrics(self._metrics(np.inf), step=1, metric_type="train")

  def test_finite_loss_does_not_exit(self):
    logger = self._make_logger(True, True)
    with (
        mock.patch.object(logger, "log_metrics"),
        mock.patch.object(logger, "write_metrics_to_tensorboard"),
        mock.patch.object(logger, "write_metrics_locally"),
        mock.patch.object(logger, "write_metrics_to_managed_mldiagnostics"),
        mock.patch("jax.process_index", return_value=1),
    ):  # skip gcs branch
      logger.write_metrics(self._metrics(1.23), step=1, metric_type="train")

  def test_abort_flags_disabled_does_not_exit(self):
    logger = self._make_logger(False, False)
    with (
        mock.patch.object(logger, "log_metrics"),
        mock.patch.object(logger, "write_metrics_to_tensorboard"),
        mock.patch.object(logger, "write_metrics_locally"),
        mock.patch.object(logger, "write_metrics_to_managed_mldiagnostics"),
        mock.patch("jax.process_index", return_value=1),
    ):
      logger.write_metrics(self._metrics(np.nan), step=1, metric_type="train")


class MetricLoggerWandbTest(unittest.TestCase):
  """Tests for MetricLogger wandb metric writing."""

  def test_write_metrics_to_wandb_flattens_scalars_and_scalars(self):
    logger = MetricLogger.__new__(MetricLogger)  # skip __init__
    metrics = {
        "scalar": {"learning/loss": 1.5},
        "scalars": {"perf": {"step_time": 0.25, "tflops_per_device": 100.0}},
    }

    fake_wandb = mock.MagicMock()
    # wandb is lazily imported inside write_metrics_to_wandb, so inject a fake module.
    with mock.patch.dict("sys.modules", {"wandb": fake_wandb}):
      logger.write_metrics_to_wandb(metrics, step=7)

    fake_wandb.log.assert_called_once_with(
        {
            "learning/loss": 1.5,
            "perf/step_time": 0.25,
            "perf/tflops_per_device": 100.0,
        },
        step=7,
    )


class MetricLoggerMetadataTest(unittest.TestCase):
  """Tests for MetricLogger metadata and setup initialization."""

  def test_metadata_init_without_tensorboard(self):
    logger = MetricLogger.__new__(MetricLogger)
    logger.config = SimpleNamespace(enable_tensorboard=False)
    logger.metadata = {}

    with (
        mock.patch("maxtext.utils.max_utils.calculate_num_params_from_pytree", return_value=1e9),
        mock.patch("maxtext.utils.maxtext_utils.calculate_tflops_training_per_device", return_value=(100.0, 0, 0)),
        mock.patch("maxtext.utils.maxtext_utils.calculate_tokens_training_per_device", return_value=1000.0),
        mock.patch("maxtext.utils.max_logging.log"),
    ):
      logger.write_setup_info_to_tensorboard({})

    self.assertEqual(logger.metadata[MetadataKey.PER_DEVICE_TFLOPS], 100.0)
    self.assertEqual(logger.metadata[MetadataKey.PER_DEVICE_TOKENS], 1000.0)


class MetricLoggerLogMetricsTest(unittest.TestCase):
  """Tests for metric_logger.log_metrics."""

  def test_log_training_metrics_elastic_enabled(self):
    logger = MetricLogger.__new__(MetricLogger)
    logger.config = SimpleNamespace(
        rampup_end_step=-1,
        hide_profiler_step_metric=False,
        num_experts=1,
        mtp_num_layers=0,
    )
    metrics = {
        "scalar": {
            "learning/loss": 1.0,
            "perf/step_time_seconds": 2.0,
            "perf/per_device_tflops_per_sec": 100.0,
            "perf/per_device_tokens_per_sec": 1000.0,
            "learning/total_weights": 100,
        }
    }

    with (
        mock.patch.object(logger, "_is_profiler_boundary_step", return_value=False),
        mock.patch("maxtext.common.metric_logger.max_logging.log") as mock_log,
        mock.patch("maxtext.common.metric_logger.elastic_utils.elastic_enabled", return_value=True),
        mock.patch("maxtext.common.metric_logger.elastic_utils.live_slice_indices", return_value=[0, 1, 2]),
    ):
      logger.log_metrics(metrics, step=1, metric_type="train")

    mock_log.assert_called_once()
    log_string = mock_log.call_args[0][0]
    self.assertIn("live slice count: 3", log_string)

  def test_log_training_metrics_elastic_disabled(self):
    logger = MetricLogger.__new__(MetricLogger)
    logger.config = SimpleNamespace(
        rampup_end_step=-1,
        hide_profiler_step_metric=False,
        num_experts=1,
        mtp_num_layers=0,
    )

    metrics = {
        "scalar": {
            "learning/loss": 1.0,
            "perf/step_time_seconds": 2.0,
            "perf/per_device_tflops_per_sec": 100.0,
            "perf/per_device_tokens_per_sec": 1000.0,
            "learning/total_weights": 100,
        }
    }

    with (
        mock.patch.object(logger, "_is_profiler_boundary_step", return_value=False),
        mock.patch("maxtext.common.metric_logger.max_logging.log") as mock_log,
        mock.patch("maxtext.common.metric_logger.elastic_utils.elastic_enabled", return_value=False),
        mock.patch("maxtext.common.metric_logger.elastic_utils.live_slice_indices") as mock_live_slices,
    ):
      logger.log_metrics(metrics, step=1, metric_type="train")

    mock_log.assert_called_once()
    log_string = mock_log.call_args[0][0]
    self.assertNotIn("live slice count", log_string)
    mock_live_slices.assert_not_called()


class MoeDroplessFallbackLogTest(unittest.TestCase):
  """Tests the log line emitted when the in-layer dropless fallback fires."""

  def _make_logger(self):
    logger = MetricLogger.__new__(MetricLogger)  # skip __init__
    logger.config = SimpleNamespace(
        rampup_end_step=0,
        hide_profiler_step_metric=False,
        num_experts=1,
        mtp_num_layers=0,
        use_indexer=False,
    )
    return logger

  def _metrics(self, fallback_layers):
    scalars = {
        "learning/loss": 1.0,
        "learning/total_weights": 1,
        "perf/step_time_seconds": 1.0,
        "perf/per_device_tflops_per_sec": 1.0,
        "perf/per_device_tokens_per_sec": 1.0,
    }
    if fallback_layers is not None:
      scalars["learning/moe_dropless_fallback_layers"] = np.int32(fallback_layers)
    return {"scalar": scalars}

  def _logged(self, fallback_layers):
    logger = self._make_logger()
    with (
        mock.patch("maxtext.common.metric_logger.elastic_utils.elastic_enabled", return_value=False),
        mock.patch("maxtext.common.metric_logger.max_logging.log") as log,
    ):
      logger._log_training_metrics(self._metrics(fallback_layers), step=7)  # pylint: disable=protected-access
    return [c.args[0] for c in log.call_args_list]

  def test_logs_when_fallback_taken(self):
    lines = self._logged(3)
    self.assertTrue(any("Step 7: MoE ragged buffer overflow detected in 3 layer(s)" in l for l in lines))

  def test_silent_when_no_fallback(self):
    self.assertFalse(any("overflow" in l for l in self._logged(0)))
    self.assertFalse(any("overflow" in l for l in self._logged(None)))

  def _eval_logged(self, per_step_fallback_layers):
    """Accumulates eval steps like the deferred flush does, then logs the eval summary."""
    logger = self._make_logger()
    logger.cumulative_eval_metrics = {"scalar": defaultdict(float)}
    for fallback_layers in per_step_fallback_layers:
      scalars = {"evaluation/total_loss": 1.0, "evaluation/total_weights": 1.0}
      if fallback_layers is not None:
        scalars["evaluation/moe_dropless_fallback_layers"] = np.int32(fallback_layers)
      logger._accumulate_eval_metrics({"scalar": scalars})  # pylint: disable=protected-access
    cumulative = logger.cumulative_eval_metrics["scalar"]
    cumulative.update({"eval/avg_loss": 1.0, "eval/avg_perplexity": 2.7})
    with mock.patch("maxtext.common.metric_logger.max_logging.log") as log:
      logger._log_eval_metrics(logger.cumulative_eval_metrics, step=5)  # pylint: disable=protected-access
    return cumulative, [c.args[0] for c in log.call_args_list]

  def test_eval_logs_fallback_summed_over_eval_steps(self):
    _, lines = self._eval_logged([2, 0, 3])
    self.assertTrue(any("Eval after train step 5: MoE ragged buffer overflow detected in 5 layer(s)" in l for l in lines))

  def test_eval_silent_when_no_fallback(self):
    for per_step in ([0, 0], [None, None]):
      with self.subTest(per_step=per_step):
        cumulative, lines = self._eval_logged(per_step)
        self.assertFalse(any("overflow" in l for l in lines))
    # Runs without layer mode get no new eval metric key.
    self.assertNotIn("eval/moe_dropless_fallback_layers", cumulative)
