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
import datetime
from types import SimpleNamespace
import unittest
from unittest import mock

from absl.testing import absltest
import jax
import jax.numpy as jnp
from maxtext.common import metric_logger as metric_logger_module
from maxtext.common.metric_logger import MetadataKey
from maxtext.common.metric_logger import MetricLogger
import numpy as np

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
        mock.patch.object(
            metric_logger_module.max_utils,
            "calculate_num_params_from_pytree",
            return_value=1e9,
        ),
        mock.patch.object(
            metric_logger_module.maxtext_utils,
            "calculate_tflops_training_per_device",
            return_value=(100.0, 0, 0),
        ),
        mock.patch.object(
            metric_logger_module.maxtext_utils,
            "calculate_tokens_training_per_device",
            return_value=1000.0,
        ),
        mock.patch.object(metric_logger_module.max_logging, "log"),
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
        mock.patch.object(metric_logger_module.max_logging, "log") as mock_log,
        mock.patch.object(
            metric_logger_module.elastic_utils,
            "elastic_enabled",
            return_value=True,
        ),
        mock.patch.object(
            metric_logger_module.elastic_utils,
            "live_slice_indices",
            return_value=[0, 1, 2],
        ),
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
        mock.patch.object(metric_logger_module.max_logging, "log") as mock_log,
        mock.patch.object(
            metric_logger_module.elastic_utils,
            "elastic_enabled",
            return_value=False,
        ),
        mock.patch.object(
            metric_logger_module.elastic_utils,
            "live_slice_indices",
        ) as mock_live_slices,
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
        mock.patch.object(
            metric_logger_module.elastic_utils,
            "elastic_enabled",
            return_value=False,
        ),
        mock.patch.object(metric_logger_module.max_logging, "log") as log,
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
    with mock.patch.object(metric_logger_module.max_logging, "log") as log:
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


class MetricLoggerHostMaterializationTest(unittest.TestCase):
  """Tests host-CPU LR evaluation and batched jax.device_get scalar reads."""

  def _make_logger(self):
    logger = MetricLogger.__new__(MetricLogger)  # skip __init__
    logger.config = SimpleNamespace(
        rampup_end_step=0,
        hide_profiler_step_metric=False,
        num_experts=1,
        mtp_num_layers=0,
        use_indexer=False,
        abort_on_nan_loss=False,
        abort_on_inf_loss=False,
        enable_tensorboard=False,
        metrics_file="",
        gcs_metrics=False,
        managed_mldiagnostics=False,
        enable_wandb=False,
        enable_mllog=False,
    )
    logger.metadata = {
        MetadataKey.PER_DEVICE_TFLOPS: 10.0,
        MetadataKey.PER_DEVICE_TOKENS: 100.0,
    }
    logger.performance_metric_queue = None
    logger.learning_rate_schedule = lambda step: jnp.float32(1e-4) * (step + 1)
    logger.start_step = 0
    logger.buffered_metrics = []
    logger.cumulative_eval_metrics = {"scalar": defaultdict(float)}
    logger._pending_eval_step_count = 0  # pylint: disable=protected-access
    return logger

  def test_record_train_metrics_evaluates_lr_on_host_cpu(self):
    logger = self._make_logger()
    try:
      cpu_devices = jax.local_devices(backend="cpu")
    except RuntimeError as err:
      self.skipTest(f"No local CPU backend available: {err}")
    if not cpu_devices:
      self.skipTest("No local CPU devices available.")
    cpu_dev = cpu_devices[0]
    metrics_step2 = {"scalar": {}}
    metrics_step3 = {"scalar": {}}
    patch_local = mock.patch("jax.local_devices", wraps=jax.local_devices)
    patch_default = mock.patch("jax.default_device", wraps=jax.default_device)
    with patch_local as mock_local_devices, patch_default as mock_default_dev:
      logger.record_train_metrics(metrics_step2, step=2, step_time=0.5)
      logger.record_train_metrics(metrics_step3, step=3, step_time=0.5)

    mock_local_devices.assert_called_once_with(backend="cpu")
    self.assertEqual(mock_default_dev.call_count, 2)
    mock_default_dev.assert_called_with(cpu_dev)
    lr_key = "learning/current_learning_rate"
    lr_step2 = metrics_step2["scalar"][lr_key]
    lr_step3 = metrics_step3["scalar"][lr_key]
    self.assertIsInstance(lr_step2, np.ndarray)
    self.assertNotIsInstance(lr_step2, jax.Array)
    self.assertAlmostEqual(lr_step2.item(), 3e-4, places=7)
    self.assertAlmostEqual(lr_step3.item(), 4e-4, places=7)

  def test_record_train_metrics_falls_back_when_no_cpu_device(self):
    no_cpu_cases = [
        (
            "runtime_error",
            {"side_effect": RuntimeError("Unknown backend: 'cpu'")},
        ),
        ("empty_list", {"return_value": []}),
    ]
    for case_name, patch_kwargs in no_cpu_cases:
      with self.subTest(case=case_name):
        logger = self._make_logger()
        logger.learning_rate_schedule = lambda step: 5e-4 * (step + 1)
        metrics_step1 = {"scalar": {}}
        metrics_step2 = {"scalar": {}}
        patch_local = mock.patch("jax.local_devices", **patch_kwargs)
        patch_log = mock.patch.object(metric_logger_module.max_logging, "log")
        with patch_local as mock_local, patch_log as mock_log:
          logger.record_train_metrics(metrics_step1, step=1, step_time=0.5)
          logger.record_train_metrics(metrics_step2, step=2, step_time=0.5)

        mock_local.assert_called_once_with(backend="cpu")
        mock_log.assert_called_once()
        self.assertIn("JAX_PLATFORMS", mock_log.call_args[0][0])
        lr_key = "learning/current_learning_rate"
        self.assertAlmostEqual(float(metrics_step1["scalar"][lr_key]), 1e-3)
        self.assertAlmostEqual(float(metrics_step2["scalar"][lr_key]), 1.5e-3)

  def test_record_train_metrics_disables_host_cpu_after_device_error(self):
    logger = self._make_logger()
    calls = []

    def flaky_schedule(step):
      calls.append(step)
      if len(calls) == 1:
        raise ValueError("Incompatible device for jitted schedule")
      return 2e-4 * (step + 1)

    logger.learning_rate_schedule = flaky_schedule
    fake_cpu = SimpleNamespace(platform="cpu")
    metrics_step1 = {"scalar": {}}
    metrics_step2 = {"scalar": {}}
    patch_local = mock.patch("jax.local_devices", return_value=[fake_cpu])
    patch_default = mock.patch(
        "jax.default_device",
        return_value=mock.MagicMock(),
    )
    patch_log = mock.patch.object(metric_logger_module.max_logging, "log")
    with patch_local, patch_default as mock_default_dev, patch_log as mock_log:
      logger.record_train_metrics(metrics_step1, step=1, step_time=0.5)
      logger.record_train_metrics(metrics_step2, step=2, step_time=0.5)

    mock_default_dev.assert_called_once_with(fake_cpu)
    mock_log.assert_called_once()
    self.assertIn("falling back to default device", mock_log.call_args[0][0])
    lr_key = "learning/current_learning_rate"
    self.assertAlmostEqual(float(metrics_step1["scalar"][lr_key]), 4e-4)
    self.assertAlmostEqual(float(metrics_step2["scalar"][lr_key]), 6e-4)

  def test_record_train_metrics_keeps_non_cpu_lr_array_for_flush(self):
    logger = self._make_logger()
    fake_tpu_arr = mock.MagicMock(spec=jax.Array)
    tpu_devices = {mock.Mock(platform="tpu")}
    fake_tpu_arr.sharding = SimpleNamespace(device_set=tpu_devices)
    logger.learning_rate_schedule = lambda step: fake_tpu_arr
    fake_cpu = SimpleNamespace(platform="cpu")
    metrics = {"scalar": {}}
    patch_local = mock.patch("jax.local_devices", return_value=[fake_cpu])
    patch_default = mock.patch(
        "jax.default_device",
        return_value=mock.MagicMock(),
    )
    patch_asarray = mock.patch.object(metric_logger_module.np, "asarray")
    with patch_local, patch_default, patch_asarray as mock_asarray:
      logger.record_train_metrics(metrics, step=1, step_time=0.5)

    mock_asarray.assert_not_called()
    lr_val = metrics["scalar"]["learning/current_learning_rate"]
    self.assertIs(lr_val, fake_tpu_arr)

  def test_flush_batches_train_device_get_and_preserves_dtypes(self):
    logger = self._make_logger()
    metrics = {
        "scalar": {
            "learning/loss": jnp.float32(1.25),
            "learning/lm_loss": jnp.bfloat16(0.5),
            "learning/total_weights": jnp.int32(4096),
            "perf/step_time_seconds": 0.5,
            "perf/per_device_tflops_per_sec": 20.0,
            "perf/per_device_tokens_per_sec": 200.0,
        },
        "scalars": {
            "perf": {"step_time": jnp.float32(0.25)},
        },
    }
    entry = ("train", 3, metrics, datetime.timedelta(seconds=0.5))

    patch_get = mock.patch("jax.device_get", wraps=jax.device_get)
    patch_log = mock.patch.object(metric_logger_module.max_logging, "log")
    patch_elastic = mock.patch.object(
        metric_logger_module.elastic_utils,
        "elastic_enabled",
        return_value=False,
    )
    patch_tracked = mock.patch.object(
        metric_logger_module.mllog_utils,
        "tracked_stats",
    )
    with (
        patch_elastic,
        patch_get as mock_device_get,
        patch_log as mock_log,
        patch_tracked as mock_tracked,
    ):
      logger._flush_one_buffered_entry(entry)  # pylint: disable=protected-access

    mock_device_get.assert_called_once()
    self.assertEqual(len(mock_device_get.call_args[0][0]), 4)
    self.assertIsInstance(metrics["scalar"]["learning/loss"], np.ndarray)
    total_weights = metrics["scalar"]["learning/total_weights"]
    self.assertEqual(total_weights.dtype, np.int32)
    step_time = metrics["scalars"]["perf"]["step_time"]
    self.assertIsInstance(step_time, np.ndarray)
    self.assertEqual(metrics["scalar"]["learning/loss"].item(), 1.25)
    log_line = mock_log.call_args[0][0]
    self.assertIn("total_weights: 4096,", log_line)
    self.assertIn("lm_loss: 0.500", log_line)
    mock_tracked.assert_called_once()

  def test_accumulate_eval_metrics_batches_device_get(self):
    logger = self._make_logger()
    eval_metrics = {
        "scalar": {
            "evaluation/total_loss": jnp.float32(6.0),
            "evaluation/total_weights": jnp.int32(3),
            "evaluation/moe_lb_loss": jnp.float32(0.2),
            "evaluation/indexer_loss": jnp.float32(0.1),
            "evaluation/mtp_loss": jnp.float32(0.3),
            "evaluation/z_loss": jnp.float32(0.05),
        }
    }
    with mock.patch("jax.device_get", wraps=jax.device_get) as mock_device_get:
      logger._accumulate_eval_metrics(eval_metrics)  # pylint: disable=protected-access

    mock_device_get.assert_called_once()
    self.assertEqual(len(mock_device_get.call_args[0][0]), 6)
    cum = logger.cumulative_eval_metrics["scalar"]
    self.assertAlmostEqual(cum["eval/total_loss"], 6.0, places=5)
    self.assertAlmostEqual(cum["eval/total_weights"], 3.0, places=5)
    self.assertAlmostEqual(cum["eval/moe_lb_loss"], 0.2, places=5)
    self.assertAlmostEqual(cum["eval/z_loss"], 0.05, places=5)


if __name__ == "__main__":
  absltest.main()
