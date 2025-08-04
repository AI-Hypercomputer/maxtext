# SPDX-License-Identifier: Apache-2.0

"""
Integration tests to check compatibility of checkpoints between different input pipelines.

These tests verify that a checkpoint saved during a training run using one
input pipeline (e.g., 'grain') can be successfully restored and continued
by a subsequent training run using a different input pipeline (e.g., 'tfds').
The tests confirm restoration by checking the starting step of the resumed runs.

Note: Make sure to run
  `bash setup_gcsfuse.sh DATASET_GCS_BUCKET=gs://maxtext-dataset MOUNT_PATH=/tmp/gcsfuse/`
before running tests locally.
"""

from datetime import datetime
import json
import os
import pytest
from MaxText.train import main as train_main
from MaxText.tests.integration_tests.checkpointing_test import get_checkpointing_command


def check_start_step(metrics_file, start_step_target):
  with open(metrics_file, "rt", encoding="utf8") as metrics:
    start_step = json.loads(metrics.readlines()[0])["step"]
  print(f"Start step is {start_step}, start step target is {start_step_target}")
  assert start_step == float(start_step_target)


def run_checkpoint_compatibility(hardware, attention_type):
  """Tests checkpoint compatibility."""

  run_date = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
  grain_command = [
      "grain_worker_count=0",
      "grain_train_files=/tmp/gcsfuse/array-record/c4/en/3.0.1/c4-train.array_record*",
  ]

  # Run training using grain input pipeline
  train_main(
      get_checkpointing_command(
          run_date,
          hardware=hardware,
          steps=1,
          metrics_file="run_1_metrics.txt",
          attention_type=attention_type,
          dataset_type="grain",
          dataset_path="/tmp/gcsfuse",
      )
      + grain_command
  )

  # Resume training using tfds input pipeline
  train_main(
      get_checkpointing_command(
          run_date,
          hardware=hardware,
          steps=2,
          metrics_file="run_2_metrics.txt",
          attention_type=attention_type,
          dataset_type="tfds",
          dataset_path="/tmp/gcsfuse",
      )
  )

  check_start_step("run_2_metrics.txt", 1.0)


@pytest.mark.integration_test
@pytest.mark.tpu_only
def test_autoselected_attention():
  run_checkpoint_compatibility("tpu", "autoselected")


@pytest.mark.integration_test
@pytest.mark.gpu_only
def test_with_dot_product():
  os.environ["NVTE_FUSED_ATTN"] = "1"  # Enable fused attention
  run_checkpoint_compatibility("gpu", "dot_product")
