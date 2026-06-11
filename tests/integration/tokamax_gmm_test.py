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

"""Test for tokamax gmm."""

import os
import tempfile
from absl.testing import absltest
from absl.testing import parameterized
import pytest
from maxtext.trainers.pre_train import train
from tests.utils.test_helpers import get_test_config_path

train_main = train.main
gettempdir = tempfile.gettempdir


@pytest.mark.integration_test
class Train(parameterized.TestCase):
  """Test for tokamax gmm."""

  @parameterized.named_parameters(
      {
          "testcase_name": "gmm bf16",
          "quantization": "",
      },
      {
          "testcase_name": "gmm fp8",
          "quantization": "fp8_full",
      },
  )
  @pytest.mark.tpu_only
  def test_different_configs(self, quantization: str):
    """Smoke train with small config."""
    test_tmpdir = os.environ.get("TEST_TMPDIR", gettempdir())
    outputs_dir = os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR", test_tmpdir)
    args = [
        None,
        get_test_config_path(),
        f"base_output_directory={test_tmpdir}",
        "run_name=toktmax_gmm_test",
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
        "sparse_matmul=True",
        "megablox=False",
        "use_tokamax_gmm=True",
        f"quantization={quantization}",
        "use_qwix_quantization=True",
        "weight_quantization_calibration_method=fixed,-224,224",
        "act_quantization_calibration_method=fixed,-224,224",
        "per_device_batch_size=2",
        "max_target_length=128",
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
