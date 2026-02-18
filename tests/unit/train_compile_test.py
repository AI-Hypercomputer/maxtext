# Copyright 2023–2026 Google LLC
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

"""Tests for the Ahead-of-Time (AOT) compilation script.

This module contains unit tests for `train_compile.py`, ensuring that various
model configurations and parallelism strategies can be successfully compiled
for different hardware topologies.
"""

import unittest
import os.path
from tempfile import gettempdir

import pytest

from MaxText.train_compile import main as train_compile_main
from tests.utils.test_helpers import get_test_config_path

pytestmark = [pytest.mark.external_training]


class TrainCompile(unittest.TestCase):
  """Tests for the Ahead of Time Compilation functionality, train_compile.py"""

  @pytest.mark.cpu_only
  def test_save_compiled_v4(self):
    temp_dir = gettempdir()
    compiled_trainstep_file = os.path.join(temp_dir, "test_compiled_v4.pickle")
    train_compile_main(
        (
            "",
            get_test_config_path(),
            f"compiled_trainstep_file={compiled_trainstep_file}",
            "compile_topology=v4-8",
            "compile_topology_num_slices=1",
            "base_emb_dim=256",
            "base_mlp_dim=256",
            "base_num_decoder_layers=2",
        )
    )

  @pytest.mark.cpu_only
  def test_save_compiled_v5e(self):
    temp_dir = gettempdir()
    compiled_trainstep_file = os.path.join(temp_dir, "test_compiled_v5e.pickle")
    train_compile_main(
        (
            "",
            get_test_config_path(),
            f"compiled_trainstep_file={compiled_trainstep_file}",
            "compile_topology=v5e-16",
            "compile_topology_num_slices=1",
            "base_emb_dim=256",
            "base_mlp_dim=256",
            "base_num_decoder_layers=2",
        )
    )

  # TODO (b/366200617) : This tests fails in AOT, but config works fine on real hardware
  @pytest.mark.skip(reason="Issue w/ kernels_test. Error: The TPU is already in use by process...")
  @pytest.mark.cpu_only
  def test_minimal_offloaded_v5e(self):
    temp_dir = gettempdir()
    compiled_trainstep_file = os.path.join(temp_dir, "test_compiled_v5e_offload.pickle")
    train_compile_main(
        (
            "",
            get_test_config_path(),
            f"compiled_trainstep_file={compiled_trainstep_file}",
            "compile_topology=v5e-256",
            "compile_topology_num_slices=1",
            "per_device_batch_size=1",
            "ici_fsdp_parallelism=16",
            "ici_tensor_parallelism=16",
            "max_target_length=2048",
            "fused_qkv=true",
            "fused_mlp=true",
            "remat_policy=minimal_offloaded",
            "use_iota_embed=true",
            "global_parameter_scale=128",
        )
    )

  @pytest.mark.cpu_only
  def test_save_flash(self):
    compiled_trainstep_file = "/tmp/test_save_flash"
    train_compile_main(
        (
            "",
            get_test_config_path(),
            f"compiled_trainstep_file={compiled_trainstep_file}",
            "compile_topology=v5p-256",
            "compile_topology_num_slices=1",
            "per_device_batch_size=1",
            "remat_policy=custom",
            "context=device",  # Context is our name for the splash attention output for both TPU and GPU kernels.
        )
    )

  @pytest.mark.cpu_only
  def test_save_compiled_v5p_two_slices(self):
    temp_dir = gettempdir()
    compiled_trainstep_file = os.path.join(temp_dir, "test_compiled_v5p_two_slices.pickle")
    train_compile_main(
        (
            "",
            get_test_config_path(),
            f"compiled_trainstep_file={compiled_trainstep_file}",
            "compile_topology=v5p-8",
            "compile_topology_num_slices=2",
            "base_emb_dim=256",
            "base_mlp_dim=256",
            "base_num_decoder_layers=2",
        )
    )

  @pytest.mark.cpu_only
  def test_save_compiled_v6e(self):
    temp_dir = gettempdir()
    compiled_trainstep_file = os.path.join(temp_dir, "test_compiled_v6e.pickle")
    train_compile_main(
        (
            "",
            get_test_config_path(),
            f"compiled_trainstep_file={compiled_trainstep_file}",
            "compile_topology=v6e-16",
            "compile_topology_num_slices=1",
            "base_emb_dim=256",
            "base_mlp_dim=256",
            "base_num_decoder_layers=2",
        )
    )

  @pytest.mark.cpu_only
  def test_save_compiled_tpu7x(self):
    temp_dir = gettempdir()
    compiled_trainstep_file = os.path.join(temp_dir, "test_compiled_tpu7x.pickle")
    train_compile_main(
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
        )
    )

  @pytest.mark.cpu_only
  def test_save_compiled_tpu7x_two_slices(self):
    temp_dir = gettempdir()
    compiled_trainstep_file = os.path.join(temp_dir, "test_compiled_tpu7x_two_slices.pickle")
    train_compile_main(
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
        )
    )

  @pytest.mark.cpu_only
  def test_sequence_parallelism(self):
    temp_dir = gettempdir()
    compiled_trainstep_file = os.path.join(temp_dir, "test_compiled.pickle")
    train_compile_main(
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
        )
    )

  @pytest.mark.cpu_only
  def test_remat_save_dot_except_mlpwi(self):
    temp_dir = gettempdir()
    compiled_trainstep_file = os.path.join(temp_dir, "test_remat_save_dot_except_mlpwi.pickle")
    train_compile_main(
        (
            "",
            get_test_config_path(),
            f"compiled_trainstep_file={compiled_trainstep_file}",
            "compile_topology=v5e-256",
            "compile_topology_num_slices=1",
            "per_device_batch_size=0.125",
            "ici_fsdp_parallelism=16",
            "ici_tensor_parallelism=16",
            "max_target_length=2048",
            "fused_qkv=true",
            "fused_mlp=true",
            "remat_policy=save_dot_except_mlpwi",
            "use_iota_embed=true",
            "global_parameter_scale=128",
        )
    )

  @pytest.mark.cpu_only
  def test_remat_save_dot_except_mlp(self):
    temp_dir = gettempdir()
    compiled_trainstep_file = os.path.join(temp_dir, "test_remat_save_dot_except_mlp.pickle")
    train_compile_main(
        (
            "",
            get_test_config_path(),
            f"compiled_trainstep_file={compiled_trainstep_file}",
            "compile_topology=v5e-256",
            "compile_topology_num_slices=1",
            "per_device_batch_size=0.25",
            "ici_fsdp_parallelism=16",
            "ici_tensor_parallelism=16",
            "max_target_length=2048",
            "fused_qkv=true",
            "fused_mlp=true",
            "remat_policy=save_dot_except_mlp",
            "use_iota_embed=true",
            "global_parameter_scale=128",
        )
    )

  @pytest.mark.cpu_only
  def test_remat_save_qkv_proj(self):
    temp_dir = gettempdir()
    compiled_trainstep_file = os.path.join(temp_dir, "test_remat_save_qkv_proj.pickle")
    train_compile_main(
        (
            "",
            get_test_config_path(),
            f"compiled_trainstep_file={compiled_trainstep_file}",
            "compile_topology=v5e-256",
            "compile_topology_num_slices=1",
            "per_device_batch_size=0.375",
            "ici_fsdp_parallelism=16",
            "ici_tensor_parallelism=16",
            "max_target_length=2048",
            "fused_qkv=true",
            "fused_mlp=true",
            "remat_policy=save_qkv_proj",
            "use_iota_embed=true",
            "global_parameter_scale=128",
        )
    )

  @pytest.mark.cpu_only
  def test_remat_full(self):
    temp_dir = gettempdir()
    compiled_trainstep_file = os.path.join(temp_dir, "test_remat_full.pickle")
    train_compile_main(
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
        )
    )

  @pytest.mark.cpu_only
  def test_custom_64x4_mesh(self):
    temp_dir = gettempdir()
    compiled_trainstep_file = os.path.join(temp_dir, "test_custom_64x4_mesh.pickle")
    train_compile_main(
        (
            "",
            get_test_config_path(),
            f"compiled_trainstep_file={compiled_trainstep_file}",
            "compile_topology=v6e-256",
            "use_iota_embed=true",
            "compile_topology_num_slices=1",
            "ici_sequence_parallelism=4",
            "global_parameter_scale=32",
            "per_device_batch_size=0.25",
            "max_target_length=65536",
            "allow_split_physical_axes=true",
            "custom_mesh=hybrid_ring_64x4",
        )
    )

  # TODO (b/376470419) : Enable when AOT test work with host offloading.
  @pytest.mark.skip(reason="Enable when AOT test work with host offloading.")
  @pytest.mark.gpu_only
  def test_llama3_1_70b_opt_offload(self):
    temp_dir = gettempdir()
    compiled_trainstep_file = os.path.join(temp_dir, "test_llama3_1_70b_opt_offload.pickle")
    train_compile_main(
        (
            "",
            get_test_config_path(),
            f"compiled_trainstep_file={compiled_trainstep_file}",
            "compile_topology=v6e-256",
            "compile_topology_num_slices=1",
            "model_name=llama3.1-70b",
            "per_device_batch_size=2",
            "optimizer_memory_host_offload=true",
            "gradient_clipping_threshold=0",
            "max_target_length=8192",
        )
    )

  @pytest.mark.cpu_only
  def test_custom_32x8_mesh(self):
    temp_dir = gettempdir()
    compiled_trainstep_file = os.path.join(temp_dir, "test_custom_32x8_mesh.pickle")
    train_compile_main(
        (
            "",
            get_test_config_path(),
            f"compiled_trainstep_file={compiled_trainstep_file}",
            "compile_topology=v6e-256",
            "use_iota_embed=true",
            "compile_topology_num_slices=1",
            "ici_expert_parallelism=8",
            "model_name=mixtral-8x7b",
            "megablox=False",
            "sparse_matmul=False",
            "capacity_factor=1",
            "per_device_batch_size=4",
            "max_target_length=1024",
            "allow_split_physical_axes=true",
            "custom_mesh=hybrid_ring_32x8",
            "attention=flash",
        )
    )

  @pytest.mark.cpu_only
  def test_moe_dropping_bf16(self):
    temp_dir = gettempdir()
    compiled_trainstep_file = os.path.join(temp_dir, "test_moe_dropping_bf16.pickle")
    train_compile_main(
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
        )
    )

  @pytest.mark.skip(reason="b/400476456 Tests are currently flaking / failing due to JAX 0.5.1 upgrade")
  @pytest.mark.cpu_only
  def test_moe_dropping_int8(self):
    temp_dir = gettempdir()
    compiled_trainstep_file = os.path.join(temp_dir, "test_moe_dropping_int8.pickle")
    train_compile_main(
        (
            "",
            get_test_config_path(),
            f"compiled_trainstep_file={compiled_trainstep_file}",
            "compile_topology=v5p-128",
            "use_iota_embed=true",
            "compile_topology_num_slices=1",
            "model_name=mixtral-8x7b",
            "sparse_matmul=False",
            "capacity_factor=1",
            "per_device_batch_size=4",
            "max_target_length=128",
            "attention=flash",
            "dtype=bfloat16",
            "quantization=int8",
        )
    )

  # TODO(b/388572320): Add int8 quantization test once this bug is fixed.
  @pytest.mark.cpu_only
  def test_moe_megablox_bf16(self):
    temp_dir = gettempdir()
    compiled_trainstep_file = os.path.join(temp_dir, "test_moe_megablox_bf16.pickle")
    train_compile_main(
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
        )
    )

  @pytest.mark.cpu_only
  def test_moe_ragged_dot_bf16(self):
    temp_dir = gettempdir()
    compiled_trainstep_file = os.path.join(temp_dir, "test_moe_ragged_dot_bf16.pickle")
    train_compile_main(
        (
            "",
            get_test_config_path(),
            f"compiled_trainstep_file={compiled_trainstep_file}",
            "compile_topology=v6e-256",
            "use_iota_embed=true",
            "compile_topology_num_slices=1",
            "model_name=mixtral-8x7b",
            "sparse_matmul=True",
            "megablox=False",
            "per_device_batch_size=4",
            "max_target_length=1024",
            "attention=flash",
            "dtype=bfloat16",
        )
    )

  @pytest.mark.cpu_only
  def test_moe_dense_bf16(self):
    temp_dir = gettempdir()
    compiled_trainstep_file = os.path.join(temp_dir, "test_moe_dense_bf16.pickle")
    train_compile_main(
        (
            "",
            get_test_config_path(),
            f"compiled_trainstep_file={compiled_trainstep_file}",
            "compile_topology=v5p-64",
            "use_iota_embed=true",
            "compile_topology_num_slices=1",
            "model_name=mixtral-8x7b",
            "sparse_matmul=False",
            "capacity_factor=-1",
            "per_device_batch_size=4",
            "max_target_length=1024",
            "attention=flash",
            "dtype=bfloat16",
        )
    )

  @pytest.mark.skip(reason="b/400476456 Tests are currently flaking / failing due to JAX 0.5.1 upgrade")
  @pytest.mark.cpu_only
  def test_moe_dense_int8(self):
    temp_dir = gettempdir()
    compiled_trainstep_file = os.path.join(temp_dir, "test_moe_dense_int8.pickle")
    train_compile_main(
        (
            "",
            get_test_config_path(),
            f"compiled_trainstep_file={compiled_trainstep_file}",
            "compile_topology=v5p-128",
            "use_iota_embed=true",
            "compile_topology_num_slices=1",
            "model_name=mixtral-8x7b",
            "sparse_matmul=False",
            "capacity_factor=-1",
            "per_device_batch_size=4",
            "max_target_length=128",
            "attention=flash",
            "dtype=bfloat16",
            "quantization=int8",
        )
    )

  @pytest.mark.cpu_only
  def test_moe_pp_bf16(self):
    temp_dir = gettempdir()
    compiled_trainstep_file = os.path.join(temp_dir, "test_moe_pp_bf16.pickle")
    train_compile_main(
        (
            "",
            get_test_config_path(),
            f"compiled_trainstep_file={compiled_trainstep_file}",
            "compile_topology=v5p-64",
            "use_iota_embed=true",
            "compile_topology_num_slices=2",
            "model_name=mixtral-8x7b",
            "sparse_matmul=False",
            "capacity_factor=1",
            "per_device_batch_size=4",
            "max_target_length=1024",
            "attention=flash",
            "dtype=bfloat16",
            "dcn_pipeline_parallelism=2",
            "num_layers_per_pipeline_stage=1",
        )
    )

  @pytest.mark.cpu_only
  def test_moe_deepseek_scanned_bf16(self):
    temp_dir = gettempdir()
    compiled_trainstep_file = os.path.join(temp_dir, "test_moe_deepseek_scanned_bf16.pickle")
    train_compile_main(
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
        )
    )

  @pytest.mark.skip(reason="Fix sharding issue of all layers of DeepSeek")
  @pytest.mark.cpu_only
  def test_moe_deepseek_unscanned_bf16(self):
    temp_dir = gettempdir()
    compiled_trainstep_file = os.path.join(temp_dir, "test_moe_deepseek_unscanned_bf16.pickle")
    train_compile_main(
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
            "per_device_batch_size=1",
            "max_target_length=1024",
            "attention=flash",
            "dtype=bfloat16",
            "weight_dtype=bfloat16",
            "scan_layers=False",
        )
    )

  @pytest.mark.cpu_only
  def test_moe_deepseek_with_device_limit(self):
    compiled_trainstep_file = "/tmp/test_moe_deepseek_with_device_limit.pickle"
    train_compile_main(
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
            "per_device_batch_size=1",
            "max_target_length=1024",
            "attention=flash",
            "dtype=bfloat16",
            "weight_dtype=bfloat16",
            "n_routing_groups=8",
            "topk_routing_group=4",
        )
    )

  @pytest.mark.cpu_only
  def test_moe_deepseek_pipeline_subset(self):
    compiled_trainstep_file = "/tmp/test_moe_deepseek_pipeline_subset.pickle"
    train_compile_main(
        (
            "",
            get_test_config_path(),
            f"compiled_trainstep_file={compiled_trainstep_file}",
            "compile_topology=v5p-64",
            "compile_topology_num_slices=8",
            "use_iota_embed=true",
            "model_name=deepseek3-test",
            "megablox=True",
            "sparse_matmul=False",
            "capacity_factor=1",
            "per_device_batch_size=1",
            "max_target_length=1024",
            "pipeline_parallel_layers=56",
            "ici_expert_parallelism=16",
            "dcn_pipeline_parallelism=8",
        )
    )

  @pytest.mark.cpu_only
  def test_pipeline_subset(self):
    compiled_trainstep_file = "/tmp/test_pipeline_subset.pickle"
    train_compile_main(
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
            "base_num_decoder_layers=61",  # Remainder of 5 will fail when sharded incorrectly.
            "ici_expert_parallelism=16",
            "dcn_pipeline_parallelism=8",
        )
    )

  @pytest.mark.cpu_only
  def test_moe_llama4_17b_16e(self):
    compiled_trainstep_file = "/tmp/test_moe_llama4_17b_16e.pickle"
    train_compile_main(
        (
            "",
            get_test_config_path(),
            f"compiled_trainstep_file={compiled_trainstep_file}",
            "compile_topology=v5p-128",
            "compile_topology_num_slices=1",
            "model_name=llama4-17b-16e",
            "per_device_batch_size=1",
            "max_target_length=1024",
            "dtype=bfloat16",
            "weight_dtype=bfloat16",
            "scan_layers=True",
            "ici_fsdp_parallelism=16",
            "ici_tensor_parallelism=4",
        )
    )

  @pytest.mark.cpu_only
  def test_moe_gpt_oss_20b_sparse_matmul(self):
    compiled_trainstep_file = "/tmp/test_moe_gpt_oss_20b_sparse_matmul.pickle"
    train_compile_main(
        (
            "",
            get_test_config_path(),
            f"compiled_trainstep_file={compiled_trainstep_file}",
            "compile_topology=v5p-16",
            "compile_topology_num_slices=1",
            "model_name=gpt-oss-20b",
            "per_device_batch_size=1",
            "max_target_length=1024",
            "dtype=bfloat16",
            "weight_dtype=bfloat16",
            "scan_layers=True",
            "sparse_matmul=True",
            "megablox=True",
            "attention=flash",
        )
    )

  @pytest.mark.cpu_only
  def test_moe_gpt_oss_20b_dense_matmul(self):
    compiled_trainstep_file = "/tmp/test_moe_gpt_oss_20b_dense_matmul.pickle"
    train_compile_main(
        (
            "",
            get_test_config_path(),
            f"compiled_trainstep_file={compiled_trainstep_file}",
            "compile_topology=v5p-16",
            "compile_topology_num_slices=1",
            "model_name=gpt-oss-20b",
            "per_device_batch_size=1",
            "max_target_length=1024",
            "dtype=bfloat16",
            "weight_dtype=bfloat16",
            "scan_layers=True",
            "sparse_matmul=False",
            "capacity_factor=-1",
            "attention=flash",
        )
    )

  @pytest.mark.cpu_only
  def test_gpt3_6b(self):
    compiled_trainstep_file = "/tmp/test_gpt3_6b"
    train_compile_main(
        (
            "",
            get_test_config_path(),
            f"compiled_trainstep_file={compiled_trainstep_file}",
            "compile_topology=v5p-256",
            "compile_topology_num_slices=1",
            "model_name=gpt3-6b",
            "per_device_batch_size=1",
        )
    )

  @pytest.mark.cpu_only
  def test_qwen3_qk_norm(self):
    """AOT test for non-llama qk norm models"""
    compiled_trainstep_file = "/tmp/test_qwen3_qk_norm"
    train_compile_main(
        (
            "",
            get_test_config_path(),
            f"compiled_trainstep_file={compiled_trainstep_file}",
            "compile_topology=v5p-8",
            "compile_topology_num_slices=1",
            "model_name=qwen3-0.6b",
            "per_device_batch_size=1",
        )
    )

  @pytest.mark.cpu_only
  def test_qwen3_next(self):
    """AOT test for qwen3-next and GatedDeltaNet implementation"""
    compiled_trainstep_file = "/tmp/test_qwen3_next"
    train_compile_main(
        (
            "",
            get_test_config_path(),
            f"compiled_trainstep_file={compiled_trainstep_file}",
            "compile_topology=v5p-256",
            "compile_topology_num_slices=1",
            "model_name=qwen3-next-80b-a3b",
            "per_device_batch_size=1",
            "max_target_length=1024",
        )
    )

  @pytest.mark.cpu_only
  def test_deepseek32(self):
    # test deepseek3.2 with sparse attention
    compiled_trainstep_file = "/tmp/test_deepseek32.pickle"
    train_compile_main(
        (
            "",
            get_test_config_path(),
            f"compiled_trainstep_file={compiled_trainstep_file}",
            "compile_topology=v5p-256",
            "use_iota_embed=true",
            "compile_topology_num_slices=1",
            "model_name=deepseek3.2-671b",
            # megablox
            "sparse_matmul=True",
            "megablox=True",
            "per_device_batch_size=1",
            "max_target_length=1024",
            "attention=flash",
            "use_tokamax_splash=True",
            "dtype=bfloat16",
            "weight_dtype=bfloat16",
            # without_device_limit
            "n_routing_groups=-1",
            "topk_routing_group=-1",
        )
    )

  @pytest.mark.cpu_only
  def test_olmo3_7b(self):
    """AOT test for Olmo3 7B implementation"""
    compiled_trainstep_file = "/tmp/test_olmo3_7b"
    train_compile_main(
        (
            "",
            get_test_config_path(),
            f"compiled_trainstep_file={compiled_trainstep_file}",
            "compile_topology=v5p-8",
            "compile_topology_num_slices=1",
            "model_name=olmo3-7b",
            "per_device_batch_size=1",
            "scan_layers=True",
            "max_target_length=1024",
        )
    )

  @pytest.mark.cpu_only
  def test_mhc_integration(self):
    """AOT test for Manifold-onstrained Hyper Connection implementation"""
    compiled_trainstep_file = "/tmp/test_mhc_integration"
    train_compile_main(
        (
            "",
            get_test_config_path(),
            f"compiled_trainstep_file={compiled_trainstep_file}",
            "compile_topology=v5p-8",
            "compile_topology_num_slices=1",
            "model_name=deepseek-custom",
            "per_device_batch_size=4",
            "scan_layers=True",
            "max_target_length=1024",
            "mhc_expansion_rate=4",
            "attention=flash",
            "use_tokamax_splash=True",
            "engram_layers=[]",
        )
    )

  @pytest.mark.cpu_only
  def test_engram_integration(self):
    """AOT test for Engram implementation"""
    compiled_trainstep_file = "/tmp/test_engram_integration"
    train_compile_main(
        (
            "",
            get_test_config_path(),
            f"compiled_trainstep_file={compiled_trainstep_file}",
            "compile_topology=v5p-8",
            "compile_topology_num_slices=1",
            "model_name=deepseek-custom",
            "per_device_batch_size=4",
            "scan_layers=False",  # TODO(ranran): update to scan_layers=True after support with Engram
            "max_target_length=1024",
            "attention=flash",
            "use_tokamax_splash=True",
            "tokenizer_type=huggingface",
            "tokenizer_path=deepseek-ai/DeepSeek-V3.2",
        )
    )
