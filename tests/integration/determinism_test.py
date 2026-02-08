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

"""Tests to verify the deterministic nature of MaxText training runs.

This module ensures that when the MaxText training is executed multiple times 
with identical configurations, the loss metrics across runs are exactly 
the same.
"""

import datetime
import json
import unittest

import pytest

from MaxText.train import main as train_main
from tests.utils.test_helpers import get_test_config_path

pytestmark = pytest.mark.integration_test


def compare_target_metrics(metrics_files, target):
  """Asserts over loss values from two runs."""
  loss = []
  for file in metrics_files:
    with open(file, "rt", encoding="utf8") as f:
      run_loss = json.loads(f.readlines()[-1])[target]
      loss.append(run_loss)
  assert loss[0] == loss[1]


class DeterminismTests(unittest.TestCase):
  """Tests determinism by running MaxText training multiple times and comparing loss."""

  @pytest.mark.tpu_only
  @pytest.mark.scheduled_only
  def test_determinism(self):
    """Executes two identical training runs and verifies training loss is the same."""
    run_name = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    common_config = [
        None,
        get_test_config_path(),
        "steps=5",
        "enable_checkpointing=False",
        "enable_data_shuffling=True",
        "enable_dropout=False",
        "base_output_directory=gs://runner-maxtext-logs",
        "dataset_path=gs://maxtext-dataset",
        "skip_jax_distributed_system=True",
    ]
    train_1_config = common_config + [
        f"run_name={run_name}_1",
        f"metrics_file={run_name}_1_metrics.txt",
    ]
    train_2_config = common_config + [
        f"run_name={run_name}_2",
        f"metrics_file={run_name}_2_metrics.txt",
    ]

    train_main(train_1_config)
    train_main(train_2_config)
    compare_target_metrics([f"{run_name}_1_metrics.txt", f"{run_name}_2_metrics.txt"], "learning/loss")
