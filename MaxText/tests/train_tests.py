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

"""Tests for train.py with various configs"""
import os
import unittest
import pytest
from train import main as train_main
from absl.testing import absltest


class TrainTests(unittest.TestCase):
  """Tests train.py with various configs"""

  CONFIGS = {
      "base": [  # short test for train.py with TFDS c4
          None,
          "configs/base.yml",
          r"base_output_directory=gs://runner-maxtext-logs",
          "run_name=runner_test",
          r"dataset_path=gs://maxtext-dataset",
          "steps=2",
          "enable_checkpointing=False",
          r"tokenizer_path=../assets/tokenizer.llama2",
      ],
      "synthetic": [  # tests base config with synthtic dataset
          None,
          "configs/base.yml",
          r"base_output_directory=gs://runner-maxtext-logs",
          "run_name=runner_test",
          r"dataset_path=gs://maxtext-dataset",
          "steps=2",
          "enable_checkpointing=False",
          "dataset_type=synthetic",
          r"tokenizer_path=../assets/tokenizer.llama2",
      ],
      "pdb_lt_1": [  # tests base config with per_device_batch_size < 1
          None,
          "configs/base.yml",
          r"base_output_directory=gs://runner-maxtext-logs",
          "run_name=runner_test",
          r"dataset_path=gs://maxtext-dataset",
          "steps=2",
          "enable_checkpointing=False",
          "per_device_batch_size=0.25",
          "ici_tensor_parallelism=4",
          r"tokenizer_path=../assets/tokenizer.llama2",
      ],
      "int8": [  # tests base config with int8
          None,
          "configs/base.yml",
          r"base_output_directory=gs://runner-maxtext-logs",
          "run_name=runner_test",
          r"dataset_path=gs://maxtext-dataset",
          "quantization=int8",
          "steps=2",
          "enable_checkpointing=False",
          r"tokenizer_path=../assets/tokenizer.llama2",
      ],
      "fp8": [  # tests base config with fp8
          None,
          "configs/base.yml",
          r"base_output_directory=gs://runner-maxtext-logs",
          "run_name=runner_test",
          r"dataset_path=gs://maxtext-dataset",
          "quantization=fp8",
          "steps=2",
          "enable_checkpointing=False",
          r"tokenizer_path=../assets/tokenizer.llama2",
      ],
      "dropout": [  # tests base config with dropout
          None,
          "configs/base.yml",
          r"base_output_directory=gs://runner-maxtext-logs",
          "run_name=runner_test",
          r"dataset_path=gs://maxtext-dataset",
          "steps=2",
          "enable_checkpointing=False",
          "max_target_length=128",
          "per_device_batch_size=1",
          "dropout_rate=0.02",
          r"tokenizer_path=../assets/tokenizer.llama2",
      ],
      "hf_input_pipeline": [  # test for train.py with TFDS c4, using HF input pipeline
          None,
          "configs/base.yml",
          r"base_output_directory=gs://runner-maxtext-logs",
          "run_name=runner_test",
          "steps=2",
          "enable_checkpointing=False",
          "dataset_type=hf",
          "hf_path=parquet",
          r"hf_train_files=gs://maxtext-dataset/hf/c4/c4-train-00000-of-01637.parquet",
          r"tokenizer_path=google-t5/t5-large",
      ],
  }

  @pytest.mark.tpu_only
  def test_tpu_base(self):
    train_main(TrainTests.CONFIGS["base"])

  @pytest.mark.gpu_only
  def test_gpu_base(self):
    train_main(TrainTests.CONFIGS["base"] + ["attention=dot_product"])

  @pytest.mark.tpu_only
  def test_tpu_synthetic(self):
    train_main(TrainTests.CONFIGS["synthetic"])

  @pytest.mark.gpu_only
  def test_gpu_synthetic(self):
    train_main(TrainTests.CONFIGS["synthetic"] + ["attention=dot_product"])

  @pytest.mark.tpu_only
  def test_tpu_pdb_lt_1(self):
    train_main(TrainTests.CONFIGS["pdb_lt_1"])

  @pytest.mark.gpu_only
  def test_gpu_pdb_lt_1(self):
    train_main(TrainTests.CONFIGS["pdb_lt_1"] + ["attention=dot_product"])

  @pytest.mark.tpu_only
  def test_tpu_int8(self):
    train_main(TrainTests.CONFIGS["int8"])

  @pytest.mark.gpu_only
  def test_gpu_int8(self):
    train_main(TrainTests.CONFIGS["int8"] + ["attention=dot_product"])

  @pytest.mark.tpu_only
  def test_tpu_fp8(self):
    train_main(TrainTests.CONFIGS["fp8"])

  @pytest.mark.gpu_only
  def test_gpu_fp8(self):
    train_main(TrainTests.CONFIGS["fp8"] + ["attention=dot_product"])

  @pytest.mark.tpu_only
  def test_tpu_dropout(self):
    train_main(TrainTests.CONFIGS["dropout"])

  @pytest.mark.gpu_only
  def test_gpu_dropout(self):
    train_main(TrainTests.CONFIGS["dropout"] + ["attention=dot_product"])

  @pytest.mark.tpu_only
  def test_tpu_hf_input_pipeline(self):
    train_main(TrainTests.CONFIGS["hf_input_pipeline"])

  @pytest.mark.gpu_only
  def test_gpu_hf_input_pipeline(self):
    train_main(TrainTests.CONFIGS["hf_input_pipeline"] + ["attention=dot_product"])

  @pytest.mark.gpu_only
  def test_gpu_cudnn_flash_te(self):
    cudnn_flash_te = [  # tests base config on GPU with flash attention"""
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
    train_main(cudnn_flash_te)


if __name__ == "__main__":
  absltest.main()
