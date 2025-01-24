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

"""Tests for decode with various configs"""
import os
import unittest
import pytest
from decode import main as decode_main
from absl.testing import absltest


class DecodeTests(unittest.TestCase):
  """Tests decode with various configs"""

  CONFIGS = {
      "base": [  # tests decode
          None,
          "configs/base.yml",
          r"base_output_directory=gs://runner-maxtext-logs",
          "run_name=runner_test",
          r"dataset_path=gs://maxtext-dataset",
          "steps=2",
          "enable_checkpointing=False",
          "ici_tensor_parallelism=4",
          "max_target_length=128",
          "per_device_batch_size=1",
          r"tokenizer_path=../assets/tokenizer.llama2",
      ],
      "int8": [  # tests decode with int8 quantization
          None,
          "configs/base.yml",
          r"base_output_directory=gs://runner-maxtext-logs",
          "run_name=runner_test",
          r"dataset_path=gs://maxtext-dataset",
          "steps=2",
          "enable_checkpointing=False",
          "ici_tensor_parallelism=4",
          "max_target_length=128",
          "per_device_batch_size=1",
          "quantization=int8",
          "quantize_kvcache=True",
          r"tokenizer_path=../assets/tokenizer.llama2",
      ],
      "pdb_lt_1": [  # tests decode with per_device_batch_size < 1
          None,
          "configs/base.yml",
          r"base_output_directory=gs://runner-maxtext-logs",
          "run_name=runner_test",
          r"dataset_path=gs://maxtext-dataset",
          "steps=2",
          "enable_checkpointing=False",
          "ici_tensor_parallelism=4",
          "max_target_length=128",
          "per_device_batch_size=.25",
          r"tokenizer_path=../assets/tokenizer.llama2",
      ],
  }

  @pytest.mark.tpu_only
  def test_tpu_base(self):
    decode_main(DecodeTests.CONFIGS["base"])

  @pytest.mark.gpu_only
  def test_gpu_base(self):
    decode_main(DecodeTests.CONFIGS["base"] + ["attention=dot_product"])

  @pytest.mark.tpu_only
  def test_tpu_int8(self):
    decode_main(DecodeTests.CONFIGS["int8"])

  @pytest.mark.gpu_only
  def test_gpu_int8(self):
    decode_main(DecodeTests.CONFIGS["int8"] + ["attention=dot_product"])

  @pytest.mark.tpu_only
  def test_tpu_pdb_lt_1(self):
    decode_main(DecodeTests.CONFIGS["pdb_lt_1"])

  @pytest.mark.gpu_only
  def test_gpu_pdb_lt_1(self):
    decode_main(DecodeTests.CONFIGS["pdb_lt_1"] + ["attention=dot_product"])


if __name__ == "__main__":
  absltest.main()
