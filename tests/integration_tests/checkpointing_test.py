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

"""
Integration tests for checkpointing functionality.

These tests verify that a training run saves a checkpoint,
and then a subsequent training run can correctly restore and
continue from that saved checkpoint.

Note: Make sure to run
  `bash tools/setup/setup_gcsfuse.sh DATASET_GCS_BUCKET=gs://maxtext-dataset MOUNT_PATH=/tmp/gcsfuse/`
before running tests locally.
"""

from datetime import datetime
import json
from math import isclose
import os.path
import pytest
from MaxText.globals import MAXTEXT_PKG_DIR
from MaxText.train import main as train_main


def get_checkpointing_command(run_date, hardware, steps, metrics_file, attention_type, dataset_type, dataset_path):
  """Generates a command list for a checkpointing test run.

  Args:
    run_date: The date of the run.
    hardware: The hardware to run on.
    steps: The number of steps to run.
    metrics_file: The file to write metrics to.
    attention_type: The type of attention to use.
    dataset_type: The type of dataset to use.
    dataset_path: The path to the dataset.

  Returns:
    A list of strings representing the command line arguments.
  """
  model_params = [
      "base_emb_dim=384",
      "base_num_query_heads=8",
      "base_num_kv_heads=8",
      "base_mlp_dim=192",
      "base_num_decoder_layers=8",
      "head_dim=128",
  ]
  pathways_command = []
  if os.getenv("JAX_PLATFORMS") == "proxy":
    pathways_command = [
        "enable_single_controller=True",
        "checkpoint_storage_use_zarr3=False",
    ]
  return (
      [
          None,
          os.path.join(MAXTEXT_PKG_DIR, "configs", "base.yml"),
          f"hardware={hardware}",
          f"run_name=runner_{run_date}",
          f"steps={steps}",
          "max_target_length=128",
          "per_device_batch_size=1",
          f"metrics_file={metrics_file}",
          "checkpoint_period=3",
          "base_output_directory=gs://runner-maxtext-logs",
          f"dataset_path={dataset_path}",
          f"dataset_type={dataset_type}",
          "async_checkpointing=False",
          f"attention={attention_type}",
      ]
      + model_params
      + pathways_command
  )


def check_loss(metrics_file, target):
  """Asserts over loss values from loaded checkpoint.

  Args:
    metrics_file: The base name of the metrics file.
    target: The target metric to check in the metrics file.
  """
  metrics_file_saved = "saved_" + metrics_file
  metrics_file_restored = "restored_" + metrics_file

  with (
      open(metrics_file_saved, "rt", encoding="utf8") as saved,
      open(metrics_file_restored, "rt", encoding="utf8") as restored,
  ):
    saved_loss = json.loads(saved.readlines()[-1])[target]
    restored_loss = json.loads(restored.readlines()[0])[target]
    # Checks that checkpoint restore was successful by comparing loss of last
    # step in saved checkpoint to loss of first step in restored checkpoint
    print("saved loss: ", saved_loss)
    print("restored loss: ", restored_loss)
    assert isclose(saved_loss, restored_loss, rel_tol=0.1)


def run_checkpointing(hardware, attention_type):
  """Tests checkpointing by saving and restoring a model.

  Args:
    hardware: The hardware to run on.
    attention_type: The type of attention to use.
  """
  run_date = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
  grain_command = [
      "grain_worker_count=0",
      "grain_train_files=/tmp/gcsfuse/array-record/c4/en/3.0.1/c4-train.array_record*",
  ]
  train_main(
      get_checkpointing_command(
          run_date,
          hardware=hardware,
          steps=1,
          metrics_file="saved_metrics.txt",
          attention_type=attention_type,
          dataset_type="grain",
          dataset_path="/tmp/gcsfuse",
      )
      + grain_command
  )

  train_main(
      get_checkpointing_command(
          run_date,
          hardware=hardware,
          steps=2,
          metrics_file="restored_metrics.txt",
          attention_type=attention_type,
          dataset_type="grain",
          dataset_path="/tmp/gcsfuse",
      )
      + grain_command
  )

  check_loss("metrics.txt", "learning/loss")


@pytest.mark.integration_test
@pytest.mark.tpu_only
def test_autoselected_attention():
  """Tests checkpointing with autoselected attention on TPU."""
  run_checkpointing("tpu", "autoselected")


@pytest.mark.integration_test
@pytest.mark.gpu_only
def test_with_dot_product():
  """Tests checkpointing with dot_product attention on GPU."""
  run_checkpointing("gpu", "dot_product")
