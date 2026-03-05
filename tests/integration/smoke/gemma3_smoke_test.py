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

""" Gemma3 Smoke test """
import os
import unittest

from absl.testing import absltest
from tests.utils.test_helpers import get_test_config_path, get_test_dataset_path, get_test_base_output_directory

from maxtext.common.gcloud_stub import is_decoupled
from maxtext.trainers.pre_train.train import main as train_main
from maxtext.utils.globals import MAXTEXT_ASSETS_ROOT


class Gemma3Train(unittest.TestCase):
  """Smoke test G3 for Gemma3"""

  def setUp(self):
    """Set up test fixtures before each test method."""
    decoupled = is_decoupled()
    self.dataset_path = get_test_dataset_path()
    self.base_output_directory = (
        os.environ.get("LOCAL_BASE_OUTPUT", get_test_base_output_directory())
        if decoupled
        else get_test_base_output_directory()
    )

  def test_gemma3_tiny_config(self):
    test_tmpdir = os.environ.get("TEST_TMPDIR")  # pylint: disable=unused-variable
    train_main(
        [
            None,
            get_test_config_path(),
            f"base_output_directory={test_tmpdir}",
            "run_name=gemma3_runner_test",
            "model_name=gemma3-4b",
            "decoder_block=gemma3",
            "base_emb_dim=256", # Small dim for speed
            "base_num_query_heads=4",
            "base_num_kv_heads=4",
            "base_mlp_dim=1024",
            "base_num_decoder_layers=6", # Gemma3 pattern is 6 layers
            "head_dim=64",
            "per_device_batch_size=1",
            "max_target_length=256",
            "dataset_type=synthetic",
            "steps=5",
            "enable_checkpointing=False",
            rf"tokenizer_path={os.path.join(MAXTEXT_ASSETS_ROOT, 'tokenizers', 'tokenizer.llama2')}", # Reuse llama tokenizer for smoke test
            "enable_goodput_recording=False",
            "enable_checkpoint_cloud_logger=False",
            "monitor_goodput=False",
            "sliding_window_size=128",
            "use_post_attn_norm=True",
            "use_post_ffw_norm=True",
            "logits_via_embedding=True",
        ]
    )

if __name__ == "__main__":
  absltest.main()
