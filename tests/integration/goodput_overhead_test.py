# Copyright 2023–2026 Google LLC
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

"""Goodput On/Off step-time overhead and loss parity regression test."""

import datetime
import json
import math
import os
import statistics
import tempfile
import unittest
import uuid

from absl.testing import absltest
from maxtext.trainers.pre_train.train import main as train_main
import pytest
from tests.utils.test_helpers import get_test_base_output_directory, get_test_config_path, get_test_dataset_path

# Number of total training steps to execute per run.
TOTAL_STEPS = 5
# Exclude compile (step 0) and warm-up (step 1) steps when computing steady-state mean step time.
WARMUP_STEPS_TO_SKIP = 2
# Maximum allowed relative increase in mean steady-state step time when Goodput is ON (2%).
MAX_ALLOWED_OVERHEAD_RATIO = 0.02


def _read_step_metrics(metrics_file: str) -> list[dict]:
  """Reads per-step JSONL metrics written by MetricLogger."""
  records = []
  with open(metrics_file, "rt", encoding="utf-8") as f:
    for line in f:
      line = line.strip()
      if line:
        records.append(json.loads(line))
  return records


class GoodputOverheadTest(unittest.TestCase):
  """Verifies that enabling Goodput recording and monitoring does not degrade step time (>2%) or alter loss."""

  @pytest.mark.integration_test
  @pytest.mark.tpu_only
  # TODO(yenke): Uncomment @pytest.mark.scheduled_only after verifying via workflow_dispatch / PR CI.
  # @pytest.mark.scheduled_only
  @pytest.mark.external_training
  def test_goodput_on_off_overhead_and_loss_parity(self):
    base_output_directory = get_test_base_output_directory()
    dataset_path = get_test_dataset_path()
    # Unique run ID per CI invocation so Cloud Logging `goodput_<run_name>` does not
    # mix or accumulate entries from previous 4-hour scheduled runs.
    run_id = f"{datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"

    with tempfile.TemporaryDirectory() as tmp_dir:
      off_metrics_file = os.path.join(tmp_dir, f"metrics_goodput_off_{run_id}.jsonl")
      on_metrics_file = os.path.join(tmp_dir, f"metrics_goodput_on_{run_id}.jsonl")

      common_args = [
          None,
          get_test_config_path(),
          f"base_output_directory={base_output_directory}",
          f"dataset_path={dataset_path}",
          "model_name=default",
          "dataset_type=synthetic",
          f"steps={TOTAL_STEPS}",
          "enable_checkpointing=False",
          "enable_tensorboard=False",
          "log_period=1",
      ]

      # 1. Baseline Run: Goodput OFF
      train_main(
          common_args
          + [
              f"run_name=goodput_overhead_off_{run_id}",
              "enable_goodput_recording=False",
              "monitor_goodput=False",
              "enable_rolling_window_goodput=False",
              "monitor_step_time_deviation=False",
              "enable_checkpoint_cloud_logger=False",
              f"metrics_file={off_metrics_file}",
          ]
      )

      # 2. Candidate Run: Goodput ON
      train_main(
          common_args
          + [
              f"run_name=goodput_overhead_on_{run_id}",
              "enable_goodput_recording=True",
              "monitor_goodput=True",
              "enable_rolling_window_goodput=True",
              "monitor_step_time_deviation=True",
              "enable_checkpoint_cloud_logger=True",
              "goodput_upload_interval_seconds=5",
              "step_deviation_interval_seconds=5",
              f"metrics_file={on_metrics_file}",
          ]
      )

      off_records = _read_step_metrics(off_metrics_file)
      on_records = _read_step_metrics(on_metrics_file)

      self.assertEqual(len(off_records), TOTAL_STEPS)
      self.assertEqual(len(on_records), TOTAL_STEPS)

      # Exclude compile (step 0) and warm-up (step 1) steps.
      off_steady_step_times = [
          float(r["perf/step_time_seconds"]) for r in off_records if int(r["step"]) >= WARMUP_STEPS_TO_SKIP
      ]
      on_steady_step_times = [
          float(r["perf/step_time_seconds"]) for r in on_records if int(r["step"]) >= WARMUP_STEPS_TO_SKIP
      ]

      self.assertTrue(len(off_steady_step_times) > 0)
      self.assertEqual(len(off_steady_step_times), len(on_steady_step_times))

      mean_off = statistics.mean(off_steady_step_times)
      mean_on = statistics.mean(on_steady_step_times)
      overhead_ratio = (mean_on - mean_off) / mean_off

      # 1. Verify final step loss parity (fault-isolation / numerical determinism check).
      final_loss_off = float(off_records[-1]["learning/loss"])
      final_loss_on = float(on_records[-1]["learning/loss"])
      self.assertTrue(
          math.isclose(final_loss_off, final_loss_on, rel_tol=1e-6, abs_tol=1e-6),
          msg=(
              f"Final loss differs between Goodput OFF ({final_loss_off:.6f}) " f"and Goodput ON ({final_loss_on:.6f})."
          ),
      )

      # 2. Verify steady-state mean step time overhead <= 2%.
      self.assertLessEqual(
          overhead_ratio,
          MAX_ALLOWED_OVERHEAD_RATIO,
          msg=(
              f"Goodput ON steady-state mean step time ({mean_on:.4f}s, steps={on_steady_step_times}) "
              f"is {overhead_ratio * 100:.2f}% slower than Goodput OFF "
              f"({mean_off:.4f}s, steps={off_steady_step_times}), exceeding the "
              f"{MAX_ALLOWED_OVERHEAD_RATIO * 100:.1f}% threshold."
          ),
      )


if __name__ == "__main__":
  absltest.main()
