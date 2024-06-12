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
from train import main as train_main
from absl.testing import absltest


class Train(unittest.TestCase):
  """Smoke test G3 only"""

  def test_tiny_config(self):
    test_tmpdir = os.environ.get("TEST_TMPDIR")
    train_main([
        None,
        "third_party/py/maxtext/configs/base.yml",
        f"base_output_directory=gs://runner-maxtext-logs",
        "run_name=runner_test",
        r"dataset_path=gs://maxtext-dataset",
        r"tokenizer_path=gs://ranran-multipod-dev/mixtral-8x7B-v0.1-Instruct/tokenizer.mistral",
        "per_device_batch_size=8",
        "max_target_length=4096",
        "dataset_type=synthetic",
        "skip_first_n_steps_for_profiler=5",
        "steps=10",
        "dtype=bfloat16",
        "weight_dtype=bfloat16",
        "enable_checkpointing=False",
        "model_name=mixtral-test",
        "ici_fsdp_parallelism=4",
        "moe_matmul=True",
        "megablox=True",
        "attention=flash",
        "profiler=xplane",
    ])


if __name__ == "__main__":
  absltest.main()
