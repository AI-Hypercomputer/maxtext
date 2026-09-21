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

"""Smoke test for sparsity.

pytest tests/integration/sparsity_test.py
"""

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
  """Smoke test for sparsity."""

  @parameterized.named_parameters(
      {
          "testcase_name": "not_quantized",
          "quantization": "",
          "use_sparsity": False,
      },
      # {
      #     "testcase_name": "fp8_full",
      #     "quantization": "fp8_full",
      #     "use_sparsity": False,
      # },
      # {
      #     "testcase_name": "fp8_full_with_sparsity",
      #     "quantization": "fp8_full",
      #     "use_sparsity": True,
      # },
  )
  @pytest.mark.tpu_only
  def test_different_quant_sparsity_configs(self, quantization: str, use_sparsity: bool):
    test_tmpdir = os.environ.get("TEST_TMPDIR", gettempdir())
    outputs_dir = os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR", test_tmpdir)
    args = [
        None,
        get_test_config_path(),
        f"base_output_directory={test_tmpdir}",
        "run_name=different_quant_sparsity_configs_test",
        # "base_emb_dim=256",
        # "base_num_query_heads=256",
        # "base_num_kv_heads=1",
        # "base_mlp_dim=256",
        # "base_moe_mlp_dim=256",
        # "base_num_decoder_layers=2",
        # "head_dim=64",
        # "decoder_block=deepseek",
        # "attention_type=mla",
        # "num_experts=4",
        # "shared_experts=1",
        # "sparse_matmul=True",
        # "megablox=False",
        "model_name=deepseek3-test",
        "base_num_decoder_layers=4",
        "override_model_config=true",
        "ici_expert_parallelism=2",
        f"quantization={quantization}",
        "use_qwix_quantization=True",
        "per_device_batch_size=2",
        "max_target_length=128",
        "dataset_type=synthetic",
        "steps=1",
        "use_tokamax_gmm=True",
        "use_gmm_v2=true",
        "enable_checkpointing=False",
        "enable_goodput_recording=False",
        "enable_checkpoint_cloud_logger=False",
        "monitor_goodput=False",
        # Toy MoE dims don't divide evenly across fsdp; loosen the sharded-params assert.
        "sharding_tolerance=0.08",
        f"metrics_file={os.path.join(outputs_dir, 'metrics.json')}",
        "retry_when_tokens_dropped=True",
        "use_ragged_sort=True",
        "ragged_buffer_factor=2",
        "use_ring_of_experts=true",
    ]
    if use_sparsity:
      args.extend(
          [
              "weight_sparsity_n=2",
              "weight_sparsity_m=4",
              "weight_sparsity_update_step=1",
          ]
      )
    train_main(args)  # pyrefly: ignore[bad-argument-type]


if __name__ == "__main__":
  absltest.main()
