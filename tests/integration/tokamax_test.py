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

"""Test for tokamax gmm and splash."""

import os
import tempfile
from absl.testing import absltest
from absl.testing import parameterized
import pytest
from maxtext.trainers.pre_train import train
from tests.utils.test_helpers import get_test_config_path, get_config_with_unique_run_name

def train_main(config_list):
  return train.main(get_config_with_unique_run_name(config_list, "tokamax_test"))

gettempdir = tempfile.gettempdir


@pytest.mark.integration_test
class Train(parameterized.TestCase):
  """Test for tokamax gmm and splash."""

  @parameterized.named_parameters(
      {
          "testcase_name": "gmm bf16",
          "quantization": "",
          "use_gmm_v2": False,
      },
      {
          "testcase_name": "gmm fp8",
          "quantization": "fp8_full",
          "use_gmm_v2": False,
      },
      {
          "testcase_name": "gmm v2 bf16",
          "quantization": "",
          "use_gmm_v2": True,
      },
  )
  @pytest.mark.tpu_only
  def test_different_configs(self, quantization: str, use_gmm_v2: bool):
    """Smoke train with small config."""
    test_tmpdir = os.environ.get("TEST_TMPDIR", gettempdir())
    outputs_dir = os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR", test_tmpdir)
    args = [
        None,
        get_test_config_path(),
        f"base_output_directory={test_tmpdir}",
        "run_name=tokamax_test",
        # model
        "base_emb_dim=256",
        "base_num_query_heads=1",
        "base_num_kv_heads=1",
        "base_mlp_dim=256",
        "base_moe_mlp_dim=256",
        "base_num_decoder_layers=2",
        "head_dim=64",
        "decoder_block=deepseek",
        "attention_type=mla",
        "num_experts=2",
        "shared_experts=1",
        # tokamax gmm
        "sparse_matmul=True",
        "megablox=False",
        "use_tokamax_gmm=True",
        f"use_gmm_v2={use_gmm_v2}",
        # tile sizes
        "wi_tile_fwd_batch_seq=128",
        "wi_tile_fwd_embed_dim=128",
        "wi_tile_fwd_mlp_dim=128",
        "wi_tile_dlhs_batch_seq=128",
        "wi_tile_dlhs_embed_dim=128",
        "wi_tile_dlhs_mlp_dim=128",
        "wi_tile_drhs_batch_seq=128",
        "wi_tile_drhs_embed_dim=128",
        "wi_tile_drhs_mlp_dim=128",
        "wo_tile_fwd_batch_seq=128",
        "wo_tile_fwd_embed_dim=128",
        "wo_tile_fwd_mlp_dim=128",
        "wo_tile_dlhs_batch_seq=128",
        "wo_tile_dlhs_embed_dim=128",
        "wo_tile_dlhs_mlp_dim=128",
        "wo_tile_drhs_batch_seq=128",
        "wo_tile_drhs_embed_dim=128",
        "wo_tile_drhs_mlp_dim=128",
        # tokamax splash
        "max_target_length=128",
        "attention=flash",
        "use_tokamax_splash=True",
        # quantization
        f"quantization={quantization}",
        f"use_qwix_quantization={quantization == 'fp8_full'}",
        "weight_quantization_calibration_method=fixed,-224,224",
        "act_quantization_calibration_method=fixed,-224,224",
        # train
        "per_device_batch_size=1",
        "dataset_type=synthetic",
        "steps=3",
        "enable_checkpointing=False",
        "enable_goodput_recording=False",
        "enable_checkpoint_cloud_logger=False",
        "monitor_goodput=False",
        f"metrics_file={os.path.join(outputs_dir, 'metrics.json')}",
    ]
    train_main(args)


if __name__ == "__main__":
  absltest.main()
