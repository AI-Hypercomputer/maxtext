# SPDX-License-Identifier: Apache-2.0

import unittest
import os.path
import pytest

from MaxText.train import main as train_main
from MaxText.globals import PKG_DIR


class SimpleDecoderLayerTest(unittest.TestCase):

  @pytest.mark.tpu_only
  def test_simple_decoder_layer(self):
    train_main(
        [
            None,
            os.path.join(PKG_DIR, "configs", "base.yml"),
            "base_output_directory=gs://runner-maxtext-logs",
            "run_name=runner_simple_decoder_layer_test",
            "dataset_path=gs://maxtext-dataset",
            "decoder_block=simple",
            "enable_checkpointing=False",
            "enable_goodput_recording=False",
            rf"tokenizer_path={os.path.join(os.path.dirname(PKG_DIR), 'assets', 'tokenizer.llama2')}",
            "steps=3",
        ]
    )

  @pytest.mark.tpu_only
  def test_mlp_decoder_layer(self):
    train_main(
        [
            None,
            os.path.join(PKG_DIR, "configs", "base.yml"),
            "base_output_directory=gs://runner-maxtext-logs",
            "run_name=runner_simple_decoder_layer_test",
            "dataset_path=gs://maxtext-dataset",
            "decoder_block=simple_mlp",
            "enable_checkpointing=False",
            "enable_goodput_recording=False",
            rf"tokenizer_path={os.path.join(os.path.dirname(PKG_DIR), 'assets', 'tokenizer.llama2')}",
            "steps=3",
        ]
    )


if __name__ == "__main__":
  unittest.main()
