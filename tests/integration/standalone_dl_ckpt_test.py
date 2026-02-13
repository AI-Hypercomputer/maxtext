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

""" Tests for the standalone_checkpointer.py """
import unittest
import pytest
from tools.gcs_benchmarks.standalone_checkpointer import main as sckpt_main
from tools.gcs_benchmarks.standalone_dataloader import main as sdl_main
from MaxText.globals import MAXTEXT_ASSETS_ROOT
from maxtext.common.gcloud_stub import is_decoupled

from datetime import datetime
import random
import string
import os
import os.path
from tests.utils.test_helpers import get_test_config_path, get_test_dataset_path, get_test_base_output_directory


class Standalone_DL_CKPT(unittest.TestCase):
  """Tests for standalone_checkpointer.py, checkpoint and restore."""

  def setUp(self):
    """Set up test fixtures before each test method."""
    decoupled = is_decoupled()
    self.dataset_path = get_test_dataset_path()
    self.base_output_directory = (
        os.environ.get("LOCAL_BASE_OUTPUT", get_test_base_output_directory())
        if decoupled
        else get_test_base_output_directory()
    )

  def _get_random_test_name(self, test_name):
    now = datetime.now()
    date_time = now.strftime("_%Y-%m-%d-%H-%M_")
    random_string = "".join(random.SystemRandom().choice(string.ascii_uppercase + string.digits) for _ in range(6))
    random_run_name = test_name + date_time + random_string
    return random_run_name

  @pytest.mark.integration_test
  @pytest.mark.tpu_only
  def test_standalone_dataloader(self):
    random_run_name = self._get_random_test_name("standalone_dataloader")
    sdl_main(
        (
            "",
            get_test_config_path(),
            f"run_name={random_run_name}",
            f"base_output_directory={self.base_output_directory}",
            f"dataset_path={self.dataset_path}",
            "steps=100",
            "enable_checkpointing=false",
            "enable_goodput_recording=False",
            rf"tokenizer_path={os.path.join(MAXTEXT_ASSETS_ROOT, 'tokenizers', 'tokenizer.llama2')}",
        )
    )  # need to pass relative path to tokenizer

  @pytest.mark.integration_test
  @pytest.mark.tpu_only
  def test_standalone_checkpointer(self):
    random_run_name = self._get_random_test_name("standalone_checkpointer")
    # checkpoint at 50
    sckpt_main(
        (
            "",
            get_test_config_path(),
            f"run_name={random_run_name}",
            f"base_output_directory={self.base_output_directory}",
            f"dataset_path={self.dataset_path}",
            "base_emb_dim=128",
            "base_num_query_heads=4",
            "base_num_kv_heads=4",
            "base_mlp_dim=128",
            "base_num_decoder_layers=2",
            "steps=60",
            "enable_checkpointing=True",
            "checkpoint_period=50",
            "async_checkpointing=False",
            "enable_goodput_recording=False",
            "skip_jax_distributed_system=True",
        )
    )
    # restore at 50 and checkpoint at 100
    sckpt_main(
        (
            "",
            get_test_config_path(),
            f"run_name={random_run_name}",
            f"base_output_directory={self.base_output_directory}",
            f"dataset_path={self.dataset_path}",
            "base_emb_dim=128",
            "base_num_query_heads=4",
            "base_num_kv_heads=4",
            "base_mlp_dim=128",
            "base_num_decoder_layers=2",
            "steps=110",
            "enable_checkpointing=True",
            "checkpoint_period=50",
            "async_checkpointing=False",
            "enable_goodput_recording=False",
            "skip_jax_distributed_system=True",
        )
    )


if __name__ == "__main__":
  unittest.main()
