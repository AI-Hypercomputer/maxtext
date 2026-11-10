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
import warnings

from absl.testing import absltest
from maxtext.trainers.pre_train.train import main as train_main
from maxtext.utils import max_logging
import pytest
from tests.utils.test_helpers import get_test_base_output_directory, get_test_config_path, get_test_dataset_path

# Number of total training steps to execute per run.
TOTAL_STEPS = 30
# Sample unprofiled steady-state steps 18–29 (12 steps), excluding compile/warm-up (steps 0–4),
# the XProf profiling window (steps 5–14), and post-profiler settling (steps 15–17).
WARMUP_STEPS_TO_SKIP = 18
# Single-run maximum empirical overhead (+0.163% on unprofiled steady-state steps 18–29).
EMPIRICAL_OVERHEAD_RATIO = 0.00163
# Soft warning multiplier (1.2x empirical max = ~0.196% overhead): emits non-blocking CI warning.
WARNING_MULTIPLIER = 1.2
SOFT_WARNING_OVERHEAD_RATIO = EMPIRICAL_OVERHEAD_RATIO * WARNING_MULTIPLIER
# Hard blocking threshold: maximum allowed relative increase in mean steady-state step time (2.0%).
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
  @pytest.mark.scheduled_only
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
          "dataset_type=synthetic",
          f"steps={TOTAL_STEPS}",
          "per_device_batch_size=12",
          "max_target_length=2048",
          "model_name=default",
          "enable_checkpointing=False",
          "log_period=10",
          "log_config=False",
          "enable_jax_profiler=True",
          "profiler=xplane",
          "skip_first_n_steps_for_profiler=5",
          "profiler_steps=10",
          "profile_cleanly=True",
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

      # Sample unprofiled steady-state steps 18–29.
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

      # 2. Level 1 (Soft Non-Blocking Warning): overhead > 1.2x empirical max (~0.196%) and <= 2.0%.
      if SOFT_WARNING_OVERHEAD_RATIO < overhead_ratio <= MAX_ALLOWED_OVERHEAD_RATIO:
        warn_msg = (
            f"Goodput ON steady-state overhead ({overhead_ratio * 100:.3f}%, "
            f"mean_on={mean_on:.4f}s vs mean_off={mean_off:.4f}s) exceeded the soft warning threshold "
            f"of {SOFT_WARNING_OVERHEAD_RATIO * 100:.3f}% (1.2x empirical max {EMPIRICAL_OVERHEAD_RATIO * 100:.3f}%), "
            f"though it remains within the {MAX_ALLOWED_OVERHEAD_RATIO * 100:.1f}% blocking limit."
        )
        max_logging.log(f"WARNING: {warn_msg}")
        warnings.warn(warn_msg, UserWarning)

      # 3. Level 2 (Hard Blocking CI Failure): steady-state mean step time overhead > 2.0%.
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
