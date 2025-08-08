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

""" Smoke test """
import os
import unittest

from absl.testing import absltest

from MaxText.train import main as train_main
from MaxText.decode import main as decode_main
from MaxText.globals import PKG_DIR


class Train(unittest.TestCase):
  """Smoke test G3 only"""

  def test_tiny_config(self):
    test_tmpdir = os.environ.get("TEST_TMPDIR")  # pylint: disable=unused-variable
    decode_main(
        [
            None,
            os.path.join(PKG_DIR, "configs", "base.yml"),
            # pylint: disable=f-string-without-interpolation
            #os.path.join(PKG_DIR, "configs", "tpu_smoke_test.yml"),
            #rf"tokenizer_path={os.path.join(os.path.dirname(PKG_DIR), 'assets', 'tokenizer.llama2')}",
            rf"tokenizer_path={os.path.join(os.path.dirname(PKG_DIR), 'assets', 'tokenizer_llama3.tiktoken')}",
            "model_name=llama3.1-8b",
            "tokenizer_type=tiktoken",
            "scan_layers=false",
            "per_device_batch_size=1",
            "ici_autoregressive_parallelism=-1",
            "ici_fsdp_parallelism=1",
            "max_prefill_predict_length=128",
            "max_target_length=256",
            "attention=dot_product",
            "skip_jax_distributed_system=True",
        ]
    )


if __name__ == "__main__":
  absltest.main()
