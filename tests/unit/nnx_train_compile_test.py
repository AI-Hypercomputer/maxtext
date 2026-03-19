# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the Ahead-of-Time (AOT) compilation script using the NNX API.

This module contains unit tests for `nnx_train_compile.py`, ensuring that
various model configurations and parallelism strategies can be successfully
compiled for different hardware topologies using the Flax NNX API.
"""

import os.path
import unittest
from tempfile import gettempdir

import pytest

from maxtext.trainers.pre_train.nnx_train_compile import main as nnx_train_compile_main
from tests.utils.test_helpers import get_test_config_path


@pytest.mark.tpu_backend
class NnxTrainCompile(unittest.TestCase):
  """Tests for the Ahead of Time Compilation functionality, nnx_train_compile.py"""

  @pytest.mark.cpu_only
  def test_save_compiled_v4(self):
    temp_dir = gettempdir()
    compiled_trainstep_file = os.path.join(temp_dir, "nnx_test_compiled_v4.pickle")
    nnx_train_compile_main(
        (
            "",
            get_test_config_path(),
            f"compiled_trainstep_file={compiled_trainstep_file}",
            "compile_topology=v4-8",
            "compile_topology_num_slices=1",
            "base_emb_dim=256",
            "base_mlp_dim=256",
            "base_num_decoder_layers=2",
            "enable_nnx=True",
            "pure_nnx_decoder=True",
        )
    )

  @pytest.mark.cpu_only
  def test_save_compiled_v5e(self):
    temp_dir = gettempdir()
    compiled_trainstep_file = os.path.join(temp_dir, "nnx_test_compiled_v5e.pickle")
    nnx_train_compile_main(
        (
            "",
            get_test_config_path(),
            f"compiled_trainstep_file={compiled_trainstep_file}",
            "compile_topology=v5e-16",
            "compile_topology_num_slices=1",
            "base_emb_dim=256",
            "base_mlp_dim=256",
            "base_num_decoder_layers=2",
            "enable_nnx=True",
            "pure_nnx_decoder=True",
        )
    )

  @pytest.mark.cpu_only
  def test_save_compiled_v5p_two_slices(self):
    temp_dir = gettempdir()
    compiled_trainstep_file = os.path.join(temp_dir, "nnx_test_compiled_v5p_two_slices.pickle")
    nnx_train_compile_main(
        (
            "",
            get_test_config_path(),
            f"compiled_trainstep_file={compiled_trainstep_file}",
            "compile_topology=v5p-8",
            "compile_topology_num_slices=2",
            "base_emb_dim=256",
            "base_mlp_dim=256",
            "base_num_decoder_layers=2",
            "enable_nnx=True",
            "pure_nnx_decoder=True",
        )
    )

  @pytest.mark.cpu_only
  def test_save_compiled_v6e(self):
    temp_dir = gettempdir()
    compiled_trainstep_file = os.path.join(temp_dir, "nnx_test_compiled_v6e.pickle")
    nnx_train_compile_main(
        (
            "",
            get_test_config_path(),
            f"compiled_trainstep_file={compiled_trainstep_file}",
            "compile_topology=v6e-16",
            "compile_topology_num_slices=1",
            "base_emb_dim=256",
            "base_mlp_dim=256",
            "base_num_decoder_layers=2",
            "enable_nnx=True",
            "pure_nnx_decoder=True",
        )
    )

  @pytest.mark.cpu_only
  def test_save_compiled_tpu7x(self):
    temp_dir = gettempdir()
    compiled_trainstep_file = os.path.join(temp_dir, "nnx_test_compiled_tpu7x.pickle")
    nnx_train_compile_main(
        (
            None,
            get_test_config_path(),
            f"compiled_trainstep_file={compiled_trainstep_file}",
            "compile_topology=tpu7x-16",
            "compile_topology_num_slices=1",
            "ici_fsdp_parallelism=16",
            "base_emb_dim=256",
            "base_mlp_dim=256",
            "base_num_decoder_layers=2",
            "enable_nnx=True",
            "pure_nnx_decoder=True",
        )
    )

  @pytest.mark.cpu_only
  def test_save_compiled_tpu7x_two_slices(self):
    temp_dir = gettempdir()
    compiled_trainstep_file = os.path.join(temp_dir, "nnx_test_compiled_tpu7x_two_slices.pickle")
    nnx_train_compile_main(
        (
            None,
            get_test_config_path(),
            f"compiled_trainstep_file={compiled_trainstep_file}",
            "compile_topology=tpu7x-8",
            "compile_topology_num_slices=2",
            "ici_fsdp_parallelism=4",
            "ici_tensor_parallelism=2",
            "dcn_data_parallelism=2",
            "base_emb_dim=256",
            "base_mlp_dim=256",
            "base_num_decoder_layers=2",
            "enable_nnx=True",
            "pure_nnx_decoder=True",
        )
    )

  @pytest.mark.cpu_only
  def test_sequence_parallelism(self):
    temp_dir = gettempdir()
    compiled_trainstep_file = os.path.join(temp_dir, "nnx_test_compiled_sequence_parallelism.pickle")
    nnx_train_compile_main(
        (
            "",
            get_test_config_path(),
            f"compiled_trainstep_file={compiled_trainstep_file}",
            "compile_topology=v5p-64",
            "use_iota_embed=true",
            "compile_topology_num_slices=1",
            "ici_sequence_parallelism=16",
            "global_parameter_scale=32",
            "per_device_batch_size=0.0625",
            "max_target_length=65536",
            "enable_nnx=True",
            "pure_nnx_decoder=True",
        )
    )

  @pytest.mark.cpu_only
  def test_remat_full(self):
    temp_dir = gettempdir()
    compiled_trainstep_file = os.path.join(temp_dir, "nnx_test_remat_full.pickle")
    nnx_train_compile_main(
        (
            "",
            get_test_config_path(),
            f"compiled_trainstep_file={compiled_trainstep_file}",
            "compile_topology=v6e-256",
            "compile_topology_num_slices=1",
            "per_device_batch_size=1",
            "ici_fsdp_parallelism=16",
            "ici_tensor_parallelism=16",
            "max_target_length=1024",
            "fused_qkv=true",
            "fused_mlp=true",
            "remat_policy=full",
            "use_iota_embed=true",
            "global_parameter_scale=128",
            "enable_nnx=True",
            "pure_nnx_decoder=True",
        )
    )

  @pytest.mark.cpu_only
  def test_save_flash(self):
    compiled_trainstep_file = "/tmp/nnx_test_save_flash"
    nnx_train_compile_main(
        (
            "",
            get_test_config_path(),
            f"compiled_trainstep_file={compiled_trainstep_file}",
            "compile_topology=v5p-256",
            "compile_topology_num_slices=1",
            "per_device_batch_size=1",
            "remat_policy=custom",
            "context=device",
            "enable_nnx=True",
            "pure_nnx_decoder=True",
        )
    )

  @pytest.mark.cpu_only
  def test_gpt3_6b(self):
    compiled_trainstep_file = "/tmp/nnx_test_gpt3_6b"
    nnx_train_compile_main(
        (
            "",
            get_test_config_path(),
            f"compiled_trainstep_file={compiled_trainstep_file}",
            "compile_topology=v5p-256",
            "compile_topology_num_slices=1",
            "model_name=gpt3-6b",
            "per_device_batch_size=1",
            "enable_nnx=True",
            "pure_nnx_decoder=True",
        )
    )

  @pytest.mark.cpu_only
  def test_moe_dropping_bf16(self):
    temp_dir = gettempdir()
    compiled_trainstep_file = os.path.join(temp_dir, "nnx_test_moe_dropping_bf16.pickle")
    nnx_train_compile_main(
        (
            "",
            get_test_config_path(),
            f"compiled_trainstep_file={compiled_trainstep_file}",
            "compile_topology=v5p-64",
            "use_iota_embed=true",
            "compile_topology_num_slices=1",
            "model_name=mixtral-8x7b",
            "sparse_matmul=False",
            "capacity_factor=1",
            "per_device_batch_size=4",
            "max_target_length=1024",
            "attention=flash",
            "dtype=bfloat16",
            "enable_nnx=True",
            "pure_nnx_decoder=True",
        )
    )

  @pytest.mark.cpu_only
  def test_moe_megablox_bf16(self):
    temp_dir = gettempdir()
    compiled_trainstep_file = os.path.join(temp_dir, "nnx_test_moe_megablox_bf16.pickle")
    nnx_train_compile_main(
        (
            "",
            get_test_config_path(),
            f"compiled_trainstep_file={compiled_trainstep_file}",
            "compile_topology=v6e-256",
            "use_iota_embed=true",
            "compile_topology_num_slices=1",
            "model_name=mixtral-8x7b",
            "sparse_matmul=True",
            "megablox=True",
            "per_device_batch_size=4",
            "max_target_length=1024",
            "attention=flash",
            "dtype=bfloat16",
            "enable_nnx=True",
            "pure_nnx_decoder=True",
        )
    )

  @pytest.mark.cpu_only
  def test_moe_deepseek_scanned_bf16(self):
    temp_dir = gettempdir()
    compiled_trainstep_file = os.path.join(temp_dir, "nnx_test_moe_deepseek_scanned_bf16.pickle")
    nnx_train_compile_main(
        (
            "",
            get_test_config_path(),
            f"compiled_trainstep_file={compiled_trainstep_file}",
            "compile_topology=v5p-64",
            "use_iota_embed=true",
            "compile_topology_num_slices=1",
            "model_name=deepseek3-test",
            "sparse_matmul=True",
            "megablox=False",
            "per_device_batch_size=2",
            "max_target_length=1024",
            "attention=flash",
            "dtype=bfloat16",
            "weight_dtype=bfloat16",
            "scan_layers=True",
            "enable_nnx=True",
            "pure_nnx_decoder=True",
        )
    )

  @pytest.mark.cpu_only
  def test_moe_megablox_ring_ep_random(self):
    temp_dir = gettempdir()
    compiled_trainstep_file = os.path.join(temp_dir, "nnx_test_moe_megablox_ring_ep_random.pickle")
    nnx_train_compile_main(
        (
            "",
            get_test_config_path(),
            f"compiled_trainstep_file={compiled_trainstep_file}",
            "compile_topology=v5p-16",
            "use_iota_embed=true",
            "compile_topology_num_slices=1",
            "model_name=deepseek3-test",
            "sparse_matmul=True",
            "megablox=True",
            "per_device_batch_size=4",
            "max_target_length=128",
            "use_ring_of_experts=True",
            "use_random_routing=True",
            "attention=flash",
            "dtype=bfloat16",
            "enable_nnx=True",
            "pure_nnx_decoder=True",
        )
    )

  @pytest.mark.cpu_only
  def test_pipeline_subset(self):
    compiled_trainstep_file = "/tmp/nnx_test_pipeline_subset.pickle"
    nnx_train_compile_main(
        (
            "",
            get_test_config_path(),
            f"compiled_trainstep_file={compiled_trainstep_file}",
            "compile_topology=v5p-128",
            "compile_topology_num_slices=8",
            "use_iota_embed=true",
            "per_device_batch_size=1",
            "max_target_length=1024",
            "pipeline_parallel_layers=56",
            "base_num_decoder_layers=61",
            "ici_expert_parallelism=16",
            "dcn_pipeline_parallelism=8",
            "enable_nnx=True",
            "pure_nnx_decoder=True",
        )
    )
