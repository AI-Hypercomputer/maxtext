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

""" Tests for the common Max Utils """
import unittest
import pytest
from train_compile import main as train_compile_main
from train import main as train_main


class TrainCompile(unittest.TestCase):
  """Tests for the Ahead of Time Compilation functionality, train_compile.py"""

  @pytest.mark.tpu_only
  def test_save_compiled_v4(self):
    compiled_trainstep_file = "/tmp/test_compiled_v4.pickle"
    train_compile_main(
        (
            None,
            "configs/base.yml",
            f"compiled_trainstep_file={compiled_trainstep_file}",
            "compile_topology=v4-8",
            "compile_topology_num_slices=1",
            "base_emb_dim=256",
            "base_mlp_dim=256",
            "base_num_decoder_layers=2",
        )
    )

  @pytest.mark.tpu_only
  def test_save_compiled_v5e(self):
    compiled_trainstep_file = "/tmp/test_compiled_v5e.pickle"
    train_compile_main(
        (
            None,
            "configs/base.yml",
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
  def test_minimal_offloaded_v5e(self):
    compiled_trainstep_file = "/tmp/test_compiled_v5e_offload.pickle"
    train_compile_main(
        (
            None,
            "configs/base.yml",
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

  @pytest.mark.tpu_only
  def test_save_compiled_v5p_two_slices(self):
    compiled_trainstep_file = "/tmp/test_compiled_v5p_two_slices.pickle"
    train_compile_main(
        (
            None,
            "configs/base.yml",
            f"compiled_trainstep_file={compiled_trainstep_file}",
            "compile_topology=v5p-8",
            "compile_topology_num_slices=2",
            "base_emb_dim=256",
            "base_mlp_dim=256",
            "base_num_decoder_layers=2",
        )
    )

  # TODO (b/374764692) : Enable when v6e AOT test when stable Jax supports v6e AOT.
  @pytest.mark.skip(reason="Enable when downstream v6e AOT support reaches stable Jax.")
  @pytest.mark.tpu_only
  def test_save_compiled_v6e(self):
    compiled_trainstep_file = "/tmp/test_compiled_v6e.pickle"
    train_compile_main(
        (
            None,
            "configs/base.yml",
            f"compiled_trainstep_file={compiled_trainstep_file}",
            "compile_topology=v6e-16",
            "compile_topology_num_slices=1",
            "base_emb_dim=256",
            "base_mlp_dim=256",
            "base_num_decoder_layers=2",
        )
    )

  @pytest.mark.tpu_only
  def test_sequence_parallelism(self):
    compiled_trainstep_file = "/tmp/test_compiled.pickle"
    train_compile_main(
        (
            None,
            "configs/base.yml",
            f"compiled_trainstep_file={compiled_trainstep_file}",
            "compile_topology=v5e-256",
            "use_iota_embed=true",
            "compile_topology_num_slices=1",
            "ici_sequence_parallelism=16",
            "global_parameter_scale=32",
            "per_device_batch_size=0.0625",
            "max_target_length=65536",
        )
    )

  @pytest.mark.tpu_only
  def test_remat_save_dot_except_mlpwi(self):
    compiled_trainstep_file = "/tmp/test_remat_save_dot_except_mlpwi.pickle"
    train_compile_main(
        (
            None,
            "configs/base.yml",
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

  @pytest.mark.tpu_only
  def test_remat_save_dot_except_mlp(self):
    compiled_trainstep_file = "/tmp/test_remat_save_dot_except_mlp.pickle"
    train_compile_main(
        (
            None,
            "configs/base.yml",
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

  @pytest.mark.tpu_only
  def test_remat_save_qkv_proj(self):
    compiled_trainstep_file = "/tmp/test_remat_save_qkv_proj.pickle"
    train_compile_main(
        (
            None,
            "configs/base.yml",
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

  @pytest.mark.tpu_only
  def test_remat_full(self):
    compiled_trainstep_file = "/tmp/test_remat_full.pickle"
    train_compile_main(
        (
            None,
            "configs/base.yml",
            f"compiled_trainstep_file={compiled_trainstep_file}",
            "compile_topology=v5e-256",
            "compile_topology_num_slices=1",
            "per_device_batch_size=1",
            "ici_fsdp_parallelism=16",
            "ici_tensor_parallelism=16",
            "max_target_length=2048",
            "fused_qkv=true",
            "fused_mlp=true",
            "remat_policy=full",
            "use_iota_embed=true",
            "global_parameter_scale=128",
        )
    )

  @pytest.mark.tpu_only
  def test_custom_64x4_mesh(self):
    compiled_trainstep_file = "/tmp/test_custom_64x4_mesh.pickle"
    train_compile_main(
        (
            None,
            "configs/base.yml",
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
  @pytest.mark.tpu_only
  def test_llama3_1_70b_opt_offload(self):
    compiled_trainstep_file = "/tmp/test_llama3_1_70b_opt_offload.pickle"
    train_compile_main(
        (
            None,
            "configs/base.yml",
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

  @pytest.mark.tpu_only
  def test_custom_32x8_mesh(self):
    compiled_trainstep_file = "/tmp/test_custom_32x8_mesh.pickle"
    train_compile_main(
        (
            None,
            "configs/base.yml",
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
