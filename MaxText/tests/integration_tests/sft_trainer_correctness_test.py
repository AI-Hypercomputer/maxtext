#  Copyright 2025 Google LLC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#       https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""Integration tests for SFT trainer correctness."""

import pytest
import subprocess


@pytest.mark.integration_test
@pytest.mark.tpu_only
def test_maxtext_with_sft_in_trl():
  command = ["gsutil", "cp", "-r", "gs://maxtext-dataset/hf/llama3.1-tokenizer", "../assets/"]
  exit_code = subprocess.call(command)
  if exit_code != 0:
    raise ValueError(f"{command} failed with exit code: {exit_code}")

  command = [
      "python3",
      "MaxText/tests/sft_trainer_correctness.py",
      "--model-name=llama3.1-8b",
      "--tokenizer-path=assets/llama3.1-tokenizer",
      "--model-ckpt-path=gs://maxtext-model-checkpoints/llama3.1-8b/2025-01-23-19-04/scanned/0/items",
  ]

  subprocess.run(command, check=True, cwd="..")
