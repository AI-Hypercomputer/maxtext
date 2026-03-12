# Copyright 2023–2025 Google LLC
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

""" Smoke test """
import os
import unittest

from absl.testing import absltest
from tests.utils.test_helpers import get_test_config_path, get_test_dataset_path, get_test_base_output_directory

from maxtext.common.gcloud_stub import is_decoupled
from maxtext.trainers.pre_train.train import main as train_main
from maxtext.utils.globals import MAXTEXT_ASSETS_ROOT


class Train(unittest.TestCase):
  """Smoke test G3 only"""

  def setUp(self):
    """Set up test fixtures before each test method."""
    decoupled = is_decoupled()
    self.dataset_path = get_test_dataset_path()
    self.base_output_directory = (
        os.environ.get("LOCAL_BASE_OUTPUT", get_test_base_output_directory())
        if decoupled
        else get_test_base_output_directory()
    )

  def test_tiny_config(self):
    test_tmpdir = os.environ.get("TEST_TMPDIR")  # pylint: disable=unused-variable
    train_main(
        [
            None,
            get_test_config_path("base.yml"),
            "model_name=deepseek3-custom-small",
            # pylint: disable=f-string-without-interpolation
            f"base_output_directory={self.base_output_directory}",
            "run_name=runner_test",
            r"dataset_path={self.dataset_path}",
            "base_emb_dim=128",
            "base_num_query_heads=4",
            "base_num_kv_heads=4",
            "base_mlp_dim=32",
            "base_num_decoder_layers=9",
            "head_dim=128",
            "attention_output_dim=64",
            "moe_model_dim=64",
            "per_device_batch_size=2",
            "max_target_length=1024",
            "dataset_type=synthetic",
            "steps=10",
            "skip_jax_distributed_system=True",
            "attention=dot_product",
            "enable_checkpointing=False",
            rf"tokenizer_path={os.path.join(MAXTEXT_ASSETS_ROOT, 'tokenizers', 'tokenizer.llama2')}",
            "enable_goodput_recording=False",
            "enable_checkpoint_cloud_logger=False",
            "monitor_goodput=False",
        ]
    )

  def test_tiny_config_deepseek_custom_scan(self):
    test_tmpdir = os.environ.get("TEST_TMPDIR")  # pylint: disable=unused-variable
    train_main(
        [
            None,
            get_test_config_path("base.yml"),
            "model_name=deepseek3-custom-small",
            "decoder_block=deepseek_custom",
            # pylint: disable=f-string-without-interpolation
            f"base_output_directory={self.base_output_directory}",
            "run_name=runner_test",
            r"dataset_path={self.dataset_path}",
            "base_emb_dim=128",
            "base_num_query_heads=4",
            "base_num_kv_heads=4",
            "base_mlp_dim=32",
            "base_num_decoder_layers=9",
            "first_num_dense_layers=1",
            "head_dim=128",
            "attention_output_dim=64",
            "moe_model_dim=64",
            "per_device_batch_size=2",
            "max_target_length=1024",
            "dataset_type=synthetic",
            "steps=10",
            "skip_jax_distributed_system=True",
            "attention=dot_product",
            "enable_checkpointing=False",
            rf"tokenizer_path={os.path.join(MAXTEXT_ASSETS_ROOT, 'tokenizers', 'tokenizer.llama2')}",
            "enable_goodput_recording=False",
            "enable_checkpoint_cloud_logger=False",
            "monitor_goodput=False",
            "scan_layers=True",
            "inhomogeneous_layer_cycle_interval=2",
            "attention_layer_hybrid_ratio=2",
        ]
    )

  def test_tiny_config_no_scan(self):
    test_tmpdir = os.environ.get("TEST_TMPDIR")  # pylint: disable=unused-variable
    train_main(
        [
            None,
            get_test_config_path(),
            "model_name=deepseek3-custom-small",
            # pylint: disable=f-string-without-interpolation
            f"base_output_directory={self.base_output_directory}",
            "run_name=runner_test",
            r"dataset_path={self.dataset_path}",
            "base_emb_dim=128",
            "base_num_query_heads=4",
            "base_num_kv_heads=4",
            "base_mlp_dim=32",
            "base_num_decoder_layers=9",
            "head_dim=128",
            "attention_output_dim=64",
            "moe_model_dim=64",
            "per_device_batch_size=2",
            "max_target_length=1024",
            "inhomogeneous_layer_cycle_interval=2",
            "attention_layer_hybrid_ratio=2",
            "dataset_type=synthetic",
            "steps=10",
            "enable_checkpointing=False",
            rf"tokenizer_path={os.path.join(MAXTEXT_ASSETS_ROOT, 'tokenizers', 'tokenizer.llama2')}",
            "enable_goodput_recording=False",
            "enable_checkpoint_cloud_logger=False",
            "monitor_goodput=False",
            "scan_layers=False",
        ]
    )

  def test_tiny_config_explicit_shardmode(self):
    test_tmpdir = os.environ.get("TEST_TMPDIR")  # pylint: disable=unused-variable
    train_main(
        [
            None,
            get_test_config_path(),
            # pylint: disable=f-string-without-interpolation
            f"base_output_directory={self.base_output_directory}",
            "run_name=runner_test",
            r"dataset_path={self.dataset_path}",
            "base_emb_dim=8",
            "base_num_query_heads=4",
            "base_num_kv_heads=4",
            "base_mlp_dim=32",
            "base_num_decoder_layers=9",
            "head_dim=128",
            "inhomogeneous_layer_cycle_interval=2",
            "attention_layer_hybrid_ratio=2",
            "per_device_batch_size=2",
            "max_target_length=1024",
            "dataset_type=synthetic",
            "steps=10",
            "shard_mode=explicit",
            "enable_checkpointing=False",
            rf"tokenizer_path={os.path.join(MAXTEXT_ASSETS_ROOT, 'tokenizers', 'tokenizer.llama2')}",
            "enable_goodput_recording=False",
            "enable_checkpoint_cloud_logger=False",
            "monitor_goodput=False",
        ]
    )

  def test_tiny_config_moe_megablox(self):
    test_tmpdir = os.environ.get("TEST_TMPDIR")  # pylint: disable=unused-variable
    train_main(
        [
            None,
            get_test_config_path(),
            "model_name=deepseek3-custom-small",
            # pylint: disable=f-string-without-interpolation
            f"base_output_directory={self.base_output_directory}",
            "run_name=runner_test",
            r"dataset_path={self.dataset_path}",
            "base_emb_dim=128",
            "base_num_query_heads=4",
            "base_num_kv_heads=4",
            "base_mlp_dim=32",
            "base_num_decoder_layers=9",
            "head_dim=128",
            "attention_output_dim=128",
            "moe_model_dim=128",
            "base_moe_mlp_dim=128",
            "shared_expert_mlp_dim=128",
            "per_device_batch_size=2",
            "max_target_length=1024",
            "dataset_type=synthetic",
            "steps=10",
            "enable_checkpointing=False",
            rf"tokenizer_path={os.path.join(MAXTEXT_ASSETS_ROOT, 'tokenizers', 'tokenizer.llama2')}",
            "enable_goodput_recording=False",
            "enable_checkpoint_cloud_logger=False",
            "monitor_goodput=False",
            "sparse_matmul=True",
            "megablox=True",
            "use_tokamax_gmm=False",
        ]
    )

  def test_tiny_config_moe_tokamax(self):
    test_tmpdir = os.environ.get("TEST_TMPDIR")  # pylint: disable=unused-variable
    train_main(
        [
            None,
            get_test_config_path(),
            "model_name=deepseek3-custom-small",
            # pylint: disable=f-string-without-interpolation
            f"base_output_directory={self.base_output_directory}",
            "run_name=runner_test",
            r"dataset_path={self.dataset_path}",
            "base_emb_dim=128",
            "base_num_query_heads=4",
            "base_num_kv_heads=4",
            "base_mlp_dim=32",
            "base_num_decoder_layers=9",
            "head_dim=128",
            "attention_output_dim=128",
            "moe_model_dim=128",
            "base_moe_mlp_dim=128",
            "shared_expert_mlp_dim=128",
            "per_device_batch_size=2",
            "max_target_length=1024",
            "inhomogeneous_layer_cycle_interval=1",
            "attention_layer_hybrid_ratio=1",
            "dataset_type=synthetic",
            "steps=10",
            "enable_checkpointing=False",
            rf"tokenizer_path={os.path.join(MAXTEXT_ASSETS_ROOT, 'tokenizers', 'tokenizer.llama2')}",
            "enable_goodput_recording=False",
            "enable_checkpoint_cloud_logger=False",
            "monitor_goodput=False",
            "sparse_matmul=True",
            "megablox=False",
            "use_tokamax_gmm=True",
        ]
    )

  def test_tiny_config_moe_vanilla(self):
    test_tmpdir = os.environ.get("TEST_TMPDIR")  # pylint: disable=unused-variable
    train_main(
        [
            None,
            get_test_config_path(),
            "model_name=deepseek3-custom-small",
            # pylint: disable=f-string-without-interpolation
            f"base_output_directory={self.base_output_directory}",
            "run_name=runner_test",
            r"dataset_path={self.dataset_path}",
            "base_emb_dim=128",
            "base_num_query_heads=4",
            "base_num_kv_heads=4",
            "base_mlp_dim=32",
            "base_num_decoder_layers=9",
            "head_dim=128",
            "attention_output_dim=128",
            "moe_model_dim=128",
            "base_moe_mlp_dim=128",
            "shared_expert_mlp_dim=128",
            "per_device_batch_size=2",
            "max_target_length=1024",
            "dataset_type=synthetic",
            "steps=10",
            "enable_checkpointing=False",
            rf"tokenizer_path={os.path.join(MAXTEXT_ASSETS_ROOT, 'tokenizers', 'tokenizer.llama2')}",
            "enable_goodput_recording=False",
            "enable_checkpoint_cloud_logger=False",
            "monitor_goodput=False",
            "sparse_matmul=True",
            "megablox=False",
            "use_tokamax_gmm=False",
        ]
    )

if __name__ == "__main__":
  absltest.main()
