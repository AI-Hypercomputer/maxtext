"""Tests for MetricLogger coverage."""

import unittest
from unittest import mock
from maxtext.common.metric_logger import MetricLogger
from maxtext.configs import pyconfig


class MetricLoggerTest(unittest.TestCase):

  def test_log_train_metrics_moe_lb_loss(self):
    config = pyconfig.initialize(
        [
            "",
            "src/maxtext/configs/base.yml",
            "run_name=test_run",
            "base_output_directory=/tmp/maxtext_output",
            "num_experts=2",
            "mtp_num_layers=0",
            "base_moe_mlp_dim=64",
            "base_mlp_dim=64",
            "skip_jax_distributed_system=True",
        ]
    )

    logger = MetricLogger(config, None)
    metrics = {
        "scalar": {
            "learning/loss": 1.0,
            "learning/lm_loss": 1.0,
            "learning/total_weights": 1000,
            "learning/moe_lb_loss": 0.000403,
            "perf/step_time_seconds": 1.0,
            "perf/per_device_tflops_per_sec": 1.0,
            "perf/per_device_tokens_per_sec": 1.0,
        }
    }
    with mock.patch("maxtext.common.metric_logger.max_logging.log") as mock_log:
      logger._log_training_metrics(metrics, 1)  # pylint: disable=protected-access
      mock_log.assert_called()
      called_args = mock_log.call_args[0][0]
      self.assertIn("moe_lb_loss: 0.000403", called_args)


if __name__ == "__main__":
  unittest.main()
