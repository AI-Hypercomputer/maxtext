# Copyright 2026 Google LLC
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

"""Integration tests for checkpoint resharding functionality.

These tests verify that a training run saves a checkpoint using one mesh topology,
and that a subsequent run can successfully restore and continue training using a
different mesh topology (resharding).
"""

from datetime import datetime
import json
from math import isclose
import pytest

from maxtext.trainers.pre_train.train import main as train_main
from tests.utils.test_helpers import (
    get_test_config_path,
    get_test_base_output_directory,
    get_test_dataset_path,
)


def get_resharding_command(run_date, steps, metrics_file, base_output_directory, dataset_path, parallelism_args):
  """Generates a command list for the resharding test run."""
  model_params = [
      "base_emb_dim=384",
      "base_num_query_heads=8",
      "base_num_kv_heads=8",
      "base_mlp_dim=192",
      "base_num_decoder_layers=8",
      "head_dim=128",
  ]

  return (
      [
          None,
          get_test_config_path(),
          f"run_name=runner_{run_date}",
          f"steps={steps}",
          f"metrics_file={metrics_file}",
          f"base_output_directory={base_output_directory}",
          f"dataset_path={dataset_path}",
          "dataset_type=synthetic",
          "grain_worker_count=0",
          "collect_stack_trace=False",
      ]
      + model_params
      + parallelism_args
  )


def check_loss(metrics_file, target):
  """Asserts that loss values match between saved and restored checkpoints.

  Verifies the resharding restoration is mathematically consistent by comparing
   the final logged loss of the initial (saved) run against the initial logged
  loss of the resumed (restored) run within a relative tolerance.
  """
  metrics_file_saved = "saved_" + metrics_file
  metrics_file_restored = "restored_" + metrics_file

  with (
      open(metrics_file_saved, "rt", encoding="utf8") as saved,
      open(metrics_file_restored, "rt", encoding="utf8") as restored,
  ):
    # Read the last line of the saved metrics to get the final pre-checkpoint loss
    saved_loss = json.loads(saved.readlines()[-1])[target]
    # Read the first line of the restored metrics to get the initial post-restoration loss
    restored_loss = json.loads(restored.readlines()[0])[target]

    print("Saved loss: ", saved_loss)
    print("Restored loss: ", restored_loss)
    # Checks that checkpoint restore was successful by comparing loss of last
    # step in saved checkpoint to loss of first step in restored checkpoint
    assert isclose(saved_loss, restored_loss, rel_tol=0.1)


@pytest.mark.integration_test
@pytest.mark.tpu_only
@pytest.mark.scheduled_only
def test_checkpoint_resharding():
  """Tests checkpoint resharding by saving and restoring with different mesh topologies."""
  run_date = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
  base_output_directory = get_test_base_output_directory()
  dataset_path = get_test_dataset_path()

  # Phase 1: Train and Save Checkpoint
  # Topology: FSDP=4, Tensor=1
  save_parallelism = [
      "checkpoint_period=10",
      "save_checkpoint_on_completion=True",  # Saves Checkpoint 0 upon job completion (model state after step 0)
      "dcn_data_parallelism=1",
      "dcn_fsdp_parallelism=1",
      "ici_fsdp_parallelism=4",
      "ici_tensor_parallelism=1",
  ]
  train_main(
      get_resharding_command(
          run_date,
          steps=1,  # Executes Step 0
          metrics_file="saved_metrics.txt",
          base_output_directory=base_output_directory,
          dataset_path=dataset_path,
          parallelism_args=save_parallelism,
      )
  )

  # Phase 2: Restore and Continue
  # Topology: FSDP=2, Tensor=2
  restore_parallelism = [
      "dcn_data_parallelism=1",
      "dcn_fsdp_parallelism=1",
      "ici_fsdp_parallelism=2",
      "ici_tensor_parallelism=2",
  ]
  train_main(
      get_resharding_command(
          run_date,
          # 'steps' defines the target global step.
          # Restores Checkpoint 0 (state after step 0), sets start_step=1, and executes Step 1 to reach global step 2.
          steps=2,
          metrics_file="restored_metrics.txt",
          base_output_directory=base_output_directory,
          dataset_path=dataset_path,
          parallelism_args=restore_parallelism,
      )
  )

  # Phase 3: Verify Loss Consistency
  check_loss("metrics.txt", "learning/loss")
