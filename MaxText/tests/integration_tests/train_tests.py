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
from MaxText.train import main as train_main
from MaxText.globals import PKG_DIR
from absl.testing import absltest


class TrainTests(unittest.TestCase):
  """Tests train.py with various configs"""

  CONFIGS = {
      "base": [  # short test for train.py with TFDS c4
          None,
          os.path.join(PKG_DIR, "configs", "base.yml"),
          "base_output_directory=gs://runner-maxtext-logs",
          "run_name=runner_test",
          "dataset_path=gs://maxtext-dataset",
          "steps=2",
          "enable_checkpointing=False",
          "enable_goodput_recording=False",
          rf"tokenizer_path={os.path.join(os.path.dirname(PKG_DIR), 'assets', 'tokenizer.llama2')}",
      ],
      "synthetic": [  # tests base config with synthtic dataset
          None,
          os.path.join(PKG_DIR, "configs", "base.yml"),
          "base_output_directory=gs://runner-maxtext-logs",
          "run_name=runner_test",
          "dataset_path=gs://maxtext-dataset",
          "steps=2",
          "enable_checkpointing=False",
          "enable_goodput_recording=False",
          "dataset_type=synthetic",
          rf"tokenizer_path={os.path.join(os.path.dirname(PKG_DIR), 'assets', 'tokenizer.llama2')}",
      ],
      "pdb_lt_1": [  # tests base config with per_device_batch_size < 1
          None,
          os.path.join(PKG_DIR, "configs", "base.yml"),
          "base_output_directory=gs://runner-maxtext-logs",
          "run_name=runner_test",
          "dataset_path=gs://maxtext-dataset",
          "steps=2",
          "enable_checkpointing=False",
          "enable_goodput_recording=False",
          "per_device_batch_size=0.25",
          "ici_tensor_parallelism=4",
          rf"tokenizer_path={os.path.join(os.path.dirname(PKG_DIR), 'assets', 'tokenizer.llama2')}",
      ],
      "tp_transpose": [  # tests base config with ici_tensor_transpose_parallelism=4
          None,
          os.path.join(PKG_DIR, "configs", "base.yml"),
          "base_output_directory=gs://runner-maxtext-logs",
          "run_name=runner_test",
          "dataset_path=gs://maxtext-dataset",
          "steps=2",
          "ici_tensor_transpose_parallelism=4",
          "enable_goodput_recording=False",
          rf"tokenizer_path={os.path.join(os.path.dirname(PKG_DIR), 'assets', 'tokenizer.llama2')}",
      ],
      "int8": [  # tests base config with int8
          None,
          os.path.join(PKG_DIR, "configs", "base.yml"),
          "base_output_directory=gs://runner-maxtext-logs",
          "run_name=runner_test",
          "dataset_path=gs://maxtext-dataset",
          "quantization=int8",
          "steps=2",
          "enable_checkpointing=False",
          "enable_goodput_recording=False",
          rf"tokenizer_path={os.path.join(os.path.dirname(PKG_DIR), 'assets', 'tokenizer.llama2')}",
      ],
      "fp8": [  # tests base config with fp8
          None,
          os.path.join(PKG_DIR, "configs", "base.yml"),
          "base_output_directory=gs://runner-maxtext-logs",
          "run_name=runner_test",
          "dataset_path=gs://maxtext-dataset",
          "quantization=fp8",
          "steps=2",
          "enable_checkpointing=False",
          "enable_goodput_recording=False",
          rf"tokenizer_path={os.path.join(os.path.dirname(PKG_DIR), 'assets', 'tokenizer.llama2')}",
      ],
      "nanoo_fp8": [  # tests base config with nanoo_fp8
          None,
          os.path.join(PKG_DIR, "configs", "base.yml"),
          "base_output_directory=gs://runner-maxtext-logs",
          "run_name=runner_test",
          "dataset_path=gs://maxtext-dataset",
          "quantization=nanoo_fp8",
          "steps=2",
          "enable_checkpointing=False",
          "enable_goodput_recording=False",
          rf"tokenizer_path={os.path.join(os.path.dirname(PKG_DIR), 'assets', 'tokenizer.llama2')}",
      ],
      "dropout": [  # tests base config with dropout
          None,
          os.path.join(PKG_DIR, "configs", "base.yml"),
          "base_output_directory=gs://runner-maxtext-logs",
          "run_name=runner_test",
          "dataset_path=gs://maxtext-dataset",
          "steps=2",
          "enable_checkpointing=False",
          "enable_goodput_recording=False",
          "max_target_length=128",
          "per_device_batch_size=1",
          "dropout_rate=0.02",
          rf"tokenizer_path={os.path.join(os.path.dirname(PKG_DIR), 'assets', 'tokenizer.llama2')}",
      ],
      "hf_input_pipeline": [  # test for train.py with TFDS c4, using HF input pipeline
          None,
          os.path.join(PKG_DIR, "configs", "base.yml"),
          "base_output_directory=gs://runner-maxtext-logs",
          "run_name=runner_test",
          "steps=2",
          "enable_checkpointing=False",
          "enable_goodput_recording=False",
          "dataset_type=hf",
          "hf_path=parquet",
          "hf_train_files=gs://maxtext-dataset/hf/c4/c4-train-00000-of-01637.parquet",
          "tokenizer_path=google-t5/t5-large",
      ],
  }

  @pytest.mark.integration_test
  @pytest.mark.tpu_only
  def test_tpu_base(self):
    train_main(TrainTests.CONFIGS["base"])

  @pytest.mark.integration_test
  @pytest.mark.gpu_only
  def test_gpu_base(self):
    train_main(TrainTests.CONFIGS["base"] + ["attention=dot_product"])

  @pytest.mark.integration_test
  @pytest.mark.tpu_only
  def test_tpu_synthetic(self):
    train_main(TrainTests.CONFIGS["synthetic"])

  @pytest.mark.integration_test
  @pytest.mark.gpu_only
  def test_gpu_synthetic(self):
    train_main(TrainTests.CONFIGS["synthetic"] + ["attention=dot_product"])

  @pytest.mark.integration_test
  @pytest.mark.tpu_only
  def test_tpu_pdb_lt_1(self):
    train_main(TrainTests.CONFIGS["pdb_lt_1"])

  @pytest.mark.integration_test
  @pytest.mark.gpu_only
  def test_gpu_pdb_lt_1(self):
    train_main(TrainTests.CONFIGS["pdb_lt_1"] + ["attention=dot_product"])

  @pytest.mark.integration_test
  @pytest.mark.tpu_only
  def test_tpu_int8(self):
    train_main(TrainTests.CONFIGS["int8"])

  @pytest.mark.integration_test
  @pytest.mark.gpu_only
  def test_gpu_int8(self):
    train_main(TrainTests.CONFIGS["int8"] + ["attention=dot_product"])

  @pytest.mark.integration_test
  @pytest.mark.tpu_only
  def test_tpu_fp8(self):
    train_main(TrainTests.CONFIGS["fp8"])

  @pytest.mark.integration_test
  @pytest.mark.gpu_only
  def test_gpu_fp8(self):
    train_main(TrainTests.CONFIGS["fp8"] + ["attention=dot_product"])

  @pytest.mark.integration_test
  @pytest.mark.gpu_only
  def test_gpu_nanoo_fp8(self):
    train_main(TrainTests.CONFIGS["nanoo_fp8"] + ["attention=dot_product"])

  @pytest.mark.integration_test
  @pytest.mark.tpu_only
  def test_tpu_dropout(self):
    train_main(TrainTests.CONFIGS["dropout"])

  @pytest.mark.integration_test
  @pytest.mark.gpu_only
  def test_gpu_dropout(self):
    train_main(TrainTests.CONFIGS["dropout"] + ["attention=dot_product"])

  @pytest.mark.integration_test
  @pytest.mark.tpu_only
  def test_tpu_hf_input_pipeline(self):
    train_main(TrainTests.CONFIGS["hf_input_pipeline"])

  @pytest.mark.integration_test
  @pytest.mark.gpu_only
  def test_gpu_hf_input_pipeline(self):
    train_main(TrainTests.CONFIGS["hf_input_pipeline"] + ["attention=dot_product"])

  @pytest.mark.integration_test
  @pytest.mark.gpu_only
  def test_gpu_cudnn_flash_te(self):
    os.environ["NVTE_FUSED_ATTN"] = "1"  # Enable fused attention
    cudnn_flash_te = [  # tests base config on GPU with flash attention"""
        None,
        os.path.join(PKG_DIR, "configs", "base.yml"),
        "base_output_directory=gs://runner-maxtext-logs",
        "run_name=runner_test",
        "dataset_path=gs://maxtext-dataset",
        "steps=2",
        "enable_checkpointing=False",
        "enable_goodput_recording=False",
        "attention=cudnn_flash_te",
        "packing=False",
        rf"tokenizer_path={os.path.join(os.path.dirname(PKG_DIR), 'assets', 'tokenizer.llama2')}",
    ]
    train_main(cudnn_flash_te)

  @pytest.mark.integration_test
  @pytest.mark.gpu_only
  def test_gpu_context_parallelism(self):
    os.environ["NVTE_FUSED_ATTN"] = "1"  # Enable fused attention
    context_parallel = [  # tests base config on GPU with context parallelism and flash attention"""
        None,
        os.path.join(PKG_DIR, "configs", "base.yml"),
        "base_output_directory=gs://runner-maxtext-logs",
        "run_name=runner_test",
        "dataset_path=gs://maxtext-dataset",
        "steps=10",
        "enable_checkpointing=False",
        "enable_goodput_recording=False",
        "attention=cudnn_flash_te",
        "ici_fsdp_parallelism=-1",
        "ici_context_parallelism=2",
        "context_parallel_load_balance=True",
        "packing=False",
        rf"tokenizer_path={os.path.join(os.path.dirname(PKG_DIR), 'assets', 'tokenizer.llama2')}",
    ]
    train_main(context_parallel)

  @pytest.mark.integration_test
  @pytest.mark.gpu_only
  def test_gpu_tensor_parallelism(self):
    os.environ["NVTE_FUSED_ATTN"] = "1"  # Enable fused attention
    tensor_parallel = [  # tests base config on GPU with context parallelism and flash attention"""
        None,
        os.path.join(PKG_DIR, "configs", "base.yml"),
        "base_output_directory=gs://runner-maxtext-logs",
        "run_name=runner_test",
        "dataset_path=gs://maxtext-dataset",
        "steps=10",
        "enable_checkpointing=False",
        "enable_goodput_recording=False",
        "attention=cudnn_flash_te",
        "ici_fsdp_parallelism=-1",
        "ici_tensor_parallelism=2",
        "packing=False",
        rf"tokenizer_path={os.path.join(os.path.dirname(PKG_DIR), 'assets', 'tokenizer.llama2')}",
    ]
    train_main(tensor_parallel)

  @pytest.mark.integration_test
  @pytest.mark.gpu_only
  def test_gpu_optimizer_offload(self):
    os.environ["NVTE_FUSED_ATTN"] = "1"  # Enable fused attention
    optimizer_offload = [  # tests base config on GPU with optimizer state offload"""
        None,
        os.path.join(PKG_DIR, "configs", "base.yml"),
        "base_output_directory=gs://runner-maxtext-logs",
        "run_name=runner_test",
        "dataset_path=gs://maxtext-dataset",
        "steps=10",
        "attention=dot_product",
        "optimizer_memory_host_offload=True",  # enable optimizer state offload
        "dataset_type=synthetic",
        "enable_checkpointing=False",
        "enable_goodput_recording=False",
        rf"tokenizer_path={os.path.join(os.path.dirname(PKG_DIR), 'assets', 'tokenizer.llama2')}",
    ]
    train_main(optimizer_offload)

  @pytest.mark.integration_test
  @pytest.mark.gpu_only
  def test_gpu_parameter_offload(self):
    os.environ["NVTE_FUSED_ATTN"] = "1"  # Enable fused attention
    parameter_offload = [  # tests base config on GPU with parameter offload"""
        None,
        os.path.join(PKG_DIR, "configs", "base.yml"),
        "base_output_directory=gs://runner-maxtext-logs",
        "run_name=runner_test",
        "dataset_path=gs://maxtext-dataset",
        "steps=10",
        "param_scan_axis=0",  # scan axis 0 is required for parameter offload
        "attention=dot_product",
        "parameter_memory_host_offload=True",  # enable parameter offload
        "dataset_type=synthetic",
        "enable_checkpointing=False",
        "enable_goodput_recording=False",
        rf"tokenizer_path={os.path.join(os.path.dirname(PKG_DIR), 'assets', 'tokenizer.llama2')}",
    ]
    train_main(parameter_offload)

  @pytest.mark.gpu_only
  def test_gpu_cudnn_flash_jax(self):
    cudnn_flash_jax = [  # tests base config on GPU with flash attention"""
        None,
        os.path.join(PKG_DIR, "configs", "base.yml"),
        "base_output_directory=gs://runner-maxtext-logs",
        "run_name=runner_test",
        "dataset_path=gs://maxtext-dataset",
        "steps=2",
        "enable_checkpointing=False",
        "enable_goodput_recording=False",
        "attention=cudnn_flash_jax",
        "packing=False",
        rf"tokenizer_path={os.path.join(os.path.dirname(PKG_DIR), 'assets', 'tokenizer.llama2')}",
    ]
    train_main(cudnn_flash_jax)

  @pytest.mark.integration_test
  @pytest.mark.tpu_only
  def test_tpu_base_model_ag_once(self):
    train_main(TrainTests.CONFIGS["base"] + ["model_fsdp_ag_once=True"])

  @pytest.mark.integration_test
  @pytest.mark.gpu_only
  def test_gpu_synthetic_model_ag_once(self):
    train_main(TrainTests.CONFIGS["synthetic"] + ["model_fsdp_ag_once=True"])


if __name__ == "__main__":
  absltest.main()
