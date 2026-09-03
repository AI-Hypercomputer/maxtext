# Copyright 2024 Google LLC
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

"""Tests for the Ahead-of-Time (AOT) compilation script in google3."""

import os
from tempfile import gettempdir
from absl.testing import absltest
from maxtext.trainers.pre_train.train_compile import main as train_compile_main
from tests.utils.test_helpers import get_test_config_path


class TrainCompileGoogleTest(absltest.TestCase):
  """Tests for train_compile.py in google3"""

  def test_compile_gf(self):
    temp_dir = gettempdir()
    compiled_trainstep_file = os.path.join(temp_dir, "test_compiled.pickle")
    train_compile_main(
        (
            "",
            get_test_config_path(),
            f"compiled_trainstep_file={compiled_trainstep_file}",
            "internal_compile=true",
            "internal_compile_num_devices=16",
            "compile_topology=gf=2x2x2",
            "compile_topology_num_slices=1",
            "per_device_batch_size=8",
            "base_emb_dim=256",
            "base_mlp_dim=256",
            "base_num_decoder_layers=2",
            "decoder_block=simple",
            "skip_jax_distributed_system=true",
        )
    )
    self.assertTrue(os.path.exists(compiled_trainstep_file))

  def test_compile_pipeline(self):
    temp_dir = gettempdir()
    compiled_trainstep_file = os.path.join(temp_dir, "test_compiled.pickle")
    train_compile_main(
        (
            "",
            get_test_config_path(),
            f"compiled_trainstep_file={compiled_trainstep_file}",
            "internal_compile=true",
            "internal_compile_num_devices=16",
            "compile_topology=gf=2x2x2",
            "compile_topology_num_slices=1",
            "per_device_batch_size=2",
            "ici_pipeline_parallelism=2",
            "ici_tensor_parallelism=2",
            "decoder_block=simple_mlp",
            "override_model_config=true",
            "base_num_decoder_layers=4",
            "pipeline_fsdp_ag_per_repeat=true",
            "skip_jax_distributed_system=true",
        )
    )
    self.assertTrue(os.path.exists(compiled_trainstep_file))

  def test_compile_qwen3_custom(self):
    temp_dir = gettempdir()
    compiled_trainstep_file = os.path.join(temp_dir, "test_compiled.pickle")
    train_compile_main(
        (
            "",
            get_test_config_path(),
            "model_name=qwen3-custom-30b-a3b",
            "override_model_config=true",
            f"compiled_trainstep_file={compiled_trainstep_file}",
            "internal_compile=true",
            "internal_compile_num_devices=16",
            "compile_topology=gf=2x2x2",
            "compile_topology_num_slices=1",
            "per_device_batch_size=8",
            "base_emb_dim=256",
            "attention_output_dim=256",
            "moe_expert_input_dim=256",
            "base_mlp_dim=256",
            "base_moe_mlp_dim=256",
            "head_dim=128",
            "base_num_query_heads=4",
            "base_num_kv_heads=4",
            "num_experts=4",  # Reduced from 128
            "num_experts_per_tok=2",  # Reduced from 8
            "base_num_decoder_layers=2",
            "max_target_length=128",
            "skip_jax_distributed_system=true",
        )
    )
    self.assertTrue(os.path.exists(compiled_trainstep_file))


if __name__ == "__main__":
  absltest.main()
