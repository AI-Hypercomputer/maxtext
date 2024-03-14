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

""" Tests for the standalone_checkpointer.py """
import unittest
import pytest
from standalone_checkpointer import main as sckpt_main
from standalone_dataloader import main as sdl_main
from datetime import datetime
import random
import string


class Standalone_DL_CKPT(unittest.TestCase):
  """Tests for standalone_checkpointer.py, checkpoint and restore. """

  def _get_random_test_name(self, test_name):
    now = datetime.now()
    date_time = now.strftime("_%Y-%m-%d-%H-%M_")
    random_string = ''.join(random.SystemRandom().choice(string.ascii_uppercase + string.digits) for _ in range(6))
    random_run_name = test_name + date_time + random_string
    return random_run_name

  @pytest.mark.tpu
  def test_standalone_dataloader(self):
    random_run_name = self._get_random_test_name("standalone_dataloader")
    sdl_main((None, "configs/base.yml", "run_name="+random_run_name, "base_output_directory=gs://runner-maxtext-logs",
            "dataset_path=gs://maxtext-dataset", "steps=100", "enable_checkpointing=false",
            "tokenizer_path=../assets/tokenizer.llama2")) # need to pass relative path to tokenizer

  @pytest.mark.tpu
  def test_standalone_checkpointer(self):
    random_run_name = self._get_random_test_name("standalone_checkpointer")
    # checkpoint at 50
    sckpt_main((None, "configs/base.yml", f"run_name={random_run_name}", "base_output_directory=gs://runner-maxtext-logs",
            "dataset_path=gs://maxtext-dataset","base_emb_dim=128", "base_num_query_heads=4", "base_num_kv_heads=4",
            "base_mlp_dim=128", "base_num_decoder_layers=2", "steps=60", "enable_checkpointing=True",
            "checkpoint_period=50", "async_checkpointing=False"))
    # restore at 50 and checkpoint at 100
    sckpt_main((None, "configs/base.yml", f"run_name={random_run_name}", "base_output_directory=gs://runner-maxtext-logs",
            "dataset_path=gs://maxtext-dataset","base_emb_dim=128", "base_num_query_heads=4", "base_num_kv_heads=4",
            "base_mlp_dim=128", "base_num_decoder_layers=2", "steps=110", "enable_checkpointing=True",
            "checkpoint_period=50", "async_checkpointing=False"))

if __name__ == '__main__':
  unittest.main()
