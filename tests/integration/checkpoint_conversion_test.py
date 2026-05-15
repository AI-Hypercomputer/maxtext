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

"""Integration test for checkpoint conversion."""

import os
import subprocess
import sys
import tempfile
import unittest
import pytest

pytestmark = [pytest.mark.integration_test]


class Qwen3CheckpointConversionTest(unittest.TestCase):
  """Tests HuggingFace to Orbax checkpoint conversion."""

  @pytest.mark.cpu_only
  def test_qwen3_30b_a3b_roundtrip_conversion(self):
    model_name = "qwen3-30b-a3b"
    base_num_decoder_layers = 2
    with tempfile.TemporaryDirectory() as tmpdir:
      to_mt_cmd = [
          sys.executable,
          "-m",
          "maxtext.checkpoint_conversion.to_maxtext",
          f"model_name={model_name}",
          f"base_output_directory={tmpdir}",
          f"base_num_decoder_layers={base_num_decoder_layers}",
          "override_model_config=True",
          "scan_layers=False",
          "hardware=cpu",
          "skip_jax_distributed_system=True",
          "checkpoint_storage_use_ocdbt=False",
          "checkpoint_storage_use_zarr3=False",
          "--save_dtype=bfloat16",
          "--lazy_load_tensors=True",
      ]
      env = os.environ.copy()
      env["JAX_PLATFORMS"] = "cpu"
      env["HF_HOME"] = tmpdir

      print("Running checkpoint conversion command:", " ".join(to_mt_cmd))
      # Inherit stdout and stderr from the parent process to stream logs in real time
      subprocess.run(to_mt_cmd, env=env, check=True)

      # Verify output directory exists and contains items
      # Output structure: tmpdir/0/items
      expected_dir = os.path.join(tmpdir, "0", "items")
      self.assertTrue(os.path.exists(expected_dir), f"Expected checkpoint directory {expected_dir} does not exist.")
      self.assertTrue(len(os.listdir(expected_dir)) > 0, f"Checkpoint directory {expected_dir} is empty.")

      # Roundtrip conversion back to HuggingFace format
      expected_hf_dir = os.path.join(tmpdir, "hf_safetensor", "qwen3-30b-a3b")
      to_hf_cmd = [
          sys.executable,
          "-m",
          "maxtext.checkpoint_conversion.to_huggingface",
          f"model_name={model_name}",
          f"load_parameters_path={expected_dir}",
          f"base_output_directory={expected_hf_dir}",
          f"base_num_decoder_layers={base_num_decoder_layers}",
          "override_model_config=True",
          "scan_layers=false",
          "weight_dtype=bfloat16",
          "hardware=cpu",
          "skip_jax_distributed_system=True",
          "--override_model_architecture=True",
      ]
      print("Running roundtrip checkpoint conversion command (to HF):", " ".join(to_hf_cmd))
      subprocess.run(to_hf_cmd, env=env, check=True)

      # Verify HF output directory exists and contains safetensors/config files
      self.assertTrue(
          os.path.exists(expected_hf_dir), f"Expected HF checkpoint directory {expected_hf_dir} does not exist."
      )
      self.assertTrue(len(os.listdir(expected_hf_dir)) > 0, f"HF checkpoint directory {expected_hf_dir} is empty.")


if __name__ == "__main__":
  unittest.main()
