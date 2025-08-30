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

import unittest
import os.path
import pytest

from MaxText.train import main as train_main
from MaxText.globals import MAXTEXT_PKG_DIR, MAXTEXT_ASSETS_ROOT


class SimpleDecoderLayerTest(unittest.TestCase):

  @pytest.mark.tpu_only
  def test_simple_decoder_layer(self):
    train_main(
        [
            None,
            os.path.join(MAXTEXT_PKG_DIR, "configs", "base.yml"),
            "base_output_directory=gs://runner-maxtext-logs",
            "run_name=runner_simple_decoder_layer_test",
            "dataset_path=gs://maxtext-dataset",
            "decoder_block=simple",
            "enable_checkpointing=False",
            "enable_goodput_recording=False",
            rf"tokenizer_path={os.path.join(MAXTEXT_ASSETS_ROOT, 'tokenizer.llama2')}",
            "steps=3",
        ]
    )

  @pytest.mark.tpu_only
  def test_mlp_decoder_layer(self):
    train_main(
        [
            None,
            os.path.join(MAXTEXT_PKG_DIR, "configs", "base.yml"),
            "base_output_directory=gs://runner-maxtext-logs",
            "run_name=runner_simple_decoder_layer_test",
            "dataset_path=gs://maxtext-dataset",
            "decoder_block=simple_mlp",
            "enable_checkpointing=False",
            "enable_goodput_recording=False",
            rf"tokenizer_path={os.path.join(MAXTEXT_ASSETS_ROOT, 'tokenizer.llama2')}",
            "steps=3",
        ]
    )


if __name__ == "__main__":
  unittest.main()
