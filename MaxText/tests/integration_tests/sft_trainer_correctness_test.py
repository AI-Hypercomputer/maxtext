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

import os.path

import pytest
import subprocess

from MaxText.globals import PKG_DIR
from MaxText.tests.sft_trainer_correctness import get_argument_parser, initialize_config, main


@pytest.mark.integration_test
@pytest.mark.tpu_only
def test_maxtext_with_sft_in_trl():
  command = [
      "gsutil",
      "cp",
      "-r",
      "gs://maxtext-dataset/hf/llama3.1-tokenizer",
      os.path.join(os.path.dirname(PKG_DIR), "assets", ""),
  ]
  exit_code = subprocess.call(command, cwd=os.path.dirname(PKG_DIR))
  if exit_code != 0:
    raise ValueError(f"{command} failed with exit code: {exit_code}")

  parser = get_argument_parser()
  test_args = parser.parse_args(
      [
          "--model-name=llama3.1-8b",
          f"--tokenizer-path={os.path.join(os.path.dirname(PKG_DIR), 'assets', 'llama3.1-tokenizer')}",
          "--model-ckpt-path=gs://maxtext-model-checkpoints/llama3.1-8b/2025-01-23-19-04/scanned/0/items",
          "--rtol=1e-05",
          "--atol=0.06",
          "--kl-div=7e-05",
      ]
  )
  config = initialize_config(test_args)
  main(config, test_args)
