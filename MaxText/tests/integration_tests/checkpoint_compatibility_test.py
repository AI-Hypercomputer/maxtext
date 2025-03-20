"""
Copyright 2024 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""Integraion tests for test_checkpointing.sh"""
from datetime import datetime
import subprocess
import pytest


def run_checkpoint_compatibility(attention_type):
  """Tests checkpoint compatibility."""

  run_date = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
  command = [
      "bash",
      "end_to_end/test_checkpoint_compatibility.sh",
      f"runner_{run_date}",  # run_name
      r"gs://runner-maxtext-logs",  # output_path
      r"gs://maxtext-dataset",  # dataset_path
      attention_type,
  ]

  subprocess.run(command, check=True, cwd="..")


@pytest.mark.integration_test
@pytest.mark.tpu_only
def test_autoselected_attention():
  run_checkpoint_compatibility("autoselected")


@pytest.mark.integration_test
@pytest.mark.gpu_only
def test_with_dot_product():
  run_checkpoint_compatibility("dot_product")
