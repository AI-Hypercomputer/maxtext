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

"""Integraion tests for test_generate_param_only_checkpoint.sh"""
from datetime import datetime
import subprocess
import pytest


def run_generate_param_only_checkpoint(attention_type, quantization):
  """Tests generating a parameter-only checkpoint."""

  run_date = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
  # fmt: off
  command = [
      "bash",
      "end_to_end/test_generate_param_only_checkpoint.sh",
      "-r", f"runner_{run_date}",
      "-o", r"gs://runner-maxtext-logs",
      "-d", r"gs://maxtext-dataset",
      "-i", "4",
      "-a", attention_type,
      "-q", quantization,
  ]

  subprocess.run(command, check=True, cwd="..")


@pytest.mark.integration_test
@pytest.mark.tpu_only
@pytest.mark.parametrize("quantization", [(""), ("int8")])
def test_autoselected_attention(quantization):
  run_generate_param_only_checkpoint("autoselected", quantization)


@pytest.mark.integration_test
@pytest.mark.gpu_only
@pytest.mark.parametrize("quantization", [(""), ("int8")])
def test_with_dot_product(quantization):
  run_generate_param_only_checkpoint("dot_product", quantization)
