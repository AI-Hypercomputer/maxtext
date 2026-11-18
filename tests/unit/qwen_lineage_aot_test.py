# Copyright 2026 Google LLC
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

"""Ahead-of-Time (XAOT) compilation tests for Qwen MAX & Qwen 3.5 Lineage recipes.

Compiles the tuned TPU7x (`tpu7x-256`, `4x4x8`, 128 chips / 256 TensorCores)
batch-split Lineage recipes for both Qwen MAX (`qwen3.8-2.4t-a95b-lineage`) and
Qwen 3.5 (`qwen3.5-397b-a17b-lineage`). Marked `scheduled_only` because full
TPU7x XLA compilation takes ~1.5–2 minutes per model.
"""

import os
import tempfile
import unittest

import pytest

from maxtext.trainers.pre_train import train_compile
from tests.utils.test_helpers import get_test_config_path


@pytest.mark.tpu_backend
@pytest.mark.scheduled_only
class QwenLineageAotTest(unittest.TestCase):
  """Scheduled XAOT compile tests for Qwen MAX & Qwen 3.5 Lineage on TPU7x."""

  def _compile_qwen_lineage_recipe(self, model_name: str, num_decoder_layers: int = 20):
    with tempfile.TemporaryDirectory() as tmpdir:
      compiled_file = os.path.join(tmpdir, f"{model_name}.pickle")
      train_compile.main((
          None,
          get_test_config_path(),
          f"model_name={model_name}",
          "override_model_config=true",
          f"compiled_trainstep_file={compiled_file}",
          "compile_topology=tpu7x-256",
          "compile_topology_num_slices=1",
          f"base_num_decoder_layers={num_decoder_layers}",
          "ici_fsdp_parallelism=16",
          "ici_expert_parallelism=8",
          "ici_parallelism=[16,8,2]",
          "per_device_batch_size=2",
          "max_target_length=4096",
          "weight_dtype=bfloat16",
          "dtype=bfloat16",
          "mu_dtype=bfloat16",
          "gradient_clipping_threshold=0.0",
          "enable_checkpointing=false",
          "enable_goodput_recording=false",
      ))
      self.assertTrue(os.path.exists(compiled_file), f"Expected compiled file at {compiled_file}")

  def test_compile_qwen_max_lineage_tpu7x_256(self):
    """AOT-compiles the tuned Qwen MAX (Qwen3.8-2.4T-A95B) 20L Lineage recipe on tpu7x-256."""
    self._compile_qwen_lineage_recipe("qwen3.8-2.4t-a95b-lineage", num_decoder_layers=20)

  def test_compile_qwen3_5_lineage_tpu7x_256(self):
    """AOT-compiles the tuned Qwen 3.5 (397B-A17B) 20L Lineage recipe on tpu7x-256."""
    self._compile_qwen_lineage_recipe("qwen3.5-397b-a17b-lineage", num_decoder_layers=20)


if __name__ == "__main__":
  unittest.main()
