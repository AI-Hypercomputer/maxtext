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

"""Smoke test for MoE using ragged_dot."""

import os
import tempfile

from absl.testing import absltest
from absl.testing import parameterized

from tests.utils.test_helpers import get_test_config_path
from MaxText import globals as train

train_main = train.main
gettempdir = tempfile.gettempdir


class Train(parameterized.TestCase):
  """Smoke test for MoE using ragged_dot in G3 only."""

  @parameterized.named_parameters(
      {"testcase_name": "not_quantized", "quantization": ""},
      {"testcase_name": "fp8_full", "quantization": "fp8_full"},
  )
  def test_tiny_config(self, quantization: str):
    test_tmpdir = os.environ.get("TEST_TMPDIR", gettempdir())
    outputs_dir = os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR", test_tmpdir)
    train_main(
        [
            None,
            get_test_config_path(),
            f"base_output_directory={test_tmpdir}",
            "run_name=ragged_dot_smoke_test",
            "base_emb_dim=128",
            "base_num_query_heads=4",
            "base_num_kv_heads=4",
            "base_mlp_dim=128",
            "base_moe_mlp_dim=128",
            "base_num_decoder_layers=8",
            "head_dim=128",
            # TODO(b/441100085): When changing the decoder_block we might
            # need to adjust the tiling.
            "decoder_block=deepseek",
            "attention_type=mla",
            "num_experts=2",
            # Enable sparse_matmul.
            "sparse_matmul=True",
            # Enable ragged_dot.
            "megablox=False",
            f'quantization="{quantization}"',
            "use_qwix_quantization=True",
            "per_device_batch_size=2",
            "max_target_length=1024",
            "dataset_type=synthetic",
            "steps=10",
            "enable_checkpointing=False",
            "enable_goodput_recording=False",
            "enable_checkpoint_cloud_logger=False",
            "monitor_goodput=False",
            f"metrics_file={os.path.join(outputs_dir, 'metrics.json')}",
        ]
    )


if __name__ == "__main__":
  absltest.main()
