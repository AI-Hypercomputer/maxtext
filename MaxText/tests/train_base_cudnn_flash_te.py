"""
Copyright 2023 Google LLC

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

"""Short test for train.py with flash attention on GPU"""
import os
import unittest
import pytest
from train import main as train_main
from absl.testing import absltest


class Train(unittest.TestCase):
  """Tests base config on GPU with flash attention"""

  @pytest.mark.gpu_only
  def test_cudnn_flash_te(self):
    train_main(
        [
            None,
            "configs/base.yml",
            r"base_output_directory=gs://runner-maxtext-logs",
            "run_name=runner_test",
            r"dataset_path=gs://maxtext-dataset",
            "steps=2",
            "enable_checkpointing=False",
            "attention=cudnn_flash_te",
            r"tokenizer_path=../assets/tokenizer.llama2",
        ]
    )


if __name__ == "__main__":
  absltest.main()
