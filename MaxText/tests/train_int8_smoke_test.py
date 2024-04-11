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

""" Smoke test for int8"""
import os
import unittest
from train import main as train_main
from absl.testing import absltest

class Train(unittest.TestCase):
  """Smoke test for int8 G3 only"""

  def test_tiny_config(self):
    test_tmpdir = os.environ.get("TEST_TMPDIR")
    train_main([None, "third_party/py/maxtext/configs/base.yml",
      f"base_output_directory=gs://runner-maxtext-logs", "run_name=runner_test",
      r"dataset_path=gs://maxtext-dataset",
      "base_emb_dim=8", "base_num_query_heads=4", "base_num_kv_heads=4", "base_mlp_dim=32",
      "base_num_decoder_layers=8", "head_dim=128", "per_device_batch_size=2",
      "max_target_length=1024", "dataset_type=synthetic", "steps=10",
      "enable_checkpointing=False", "quantization=int8"])

if __name__ == '__main__':
  absltest.main()
