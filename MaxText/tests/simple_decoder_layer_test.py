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

import unittest
import pytest
from train import main as train_main


class SimpleDecoderLayerTest(unittest.TestCase):
  @pytest.mark.tpu
  def test_simple_decoder_layer(self):
    train_main([
          None,
          "configs/base.yml",
          r"base_output_directory=gs://runner-maxtext-logs",
          "run_name=runner_simple_decoder_layer_test",
          r"dataset_path=gs://maxtext-dataset",
          "decoder_block=simple",
          "enable_checkpointing=False",
          "tokenizer_path=../assets/tokenizer.llama2",
    ])

if __name__ == "__main__":
  unittest.main()
