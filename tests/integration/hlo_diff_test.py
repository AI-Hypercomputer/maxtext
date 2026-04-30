# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for HLO Graph Diff Verification.

Integrates validation checks system that detects unintended compiler graph transformations
or model graph deviations without breaking isolated PR constraints validations.

This is part 1 of the automated compiler validation pipeline:
- Ensures the PR isn't changing compiler performance and model graph generation unintentionally.
- Checks current runtime dumps against base reference checkpoints (in tests/utils/reference_hlo_*.txt).
"""

import os
import unittest
import shutil
import jax
import pytest
from maxtext.trainers.pre_train import train_compile
from absl import flags
import difflib
import re

pytestmark = [pytest.mark.integration_test]

# Maximum number of filtered lines from HLO to compare or store in the reference file
MAX_LINES = 2000

# Matches operation sections content like: 1234 {"sharding parameters", ...} or 1234 "operation metadata"
SECTION_CONTENT_REGEX = re.compile(r'^\d+ (?:"[^"]*"|\{[^\}]*\})$')


# Filters and normalizes an HLO line to ignore environment-specific details.
def filter_line(line):
  # Matches operation sections content like: 1234 {"sharding parameters", ...} or 1234 "operation metadata"
  if SECTION_CONTENT_REGEX.match(line):
    return None
  return line


@pytest.mark.tpu_backend
class TestHloDiff:
  """Tests for HLO Graph Diff Verification."""

  def setup_method(self):
    # Disable cache to ensure compilation occurs every time
    jax.config.update("jax_enable_compilation_cache", False)

  def check_files_equal_ignoring_sections(self, file_path1, file_path2):
    """Asserts that two text files are identical, ignoring specific sections."""

    def get_filtered_lines(file_path):
      with open(file_path, "r", encoding="utf-8") as f:
        lines = []
        for line in f:
          line = filter_line(line)
          if line is None:
            continue
          lines.append(line)
        return lines

    lines1 = get_filtered_lines(file_path1)[:MAX_LINES]
    lines2 = get_filtered_lines(file_path2)[:MAX_LINES]

    if lines1 == lines2:
      return True

    print("\n" + "=" * 20 + " HLO Diff " + "=" * 20)
    for line in difflib.unified_diff(lines1, lines2, fromfile=file_path1, tofile=file_path2):
      print(line, end="")
    print("=" * 50 + "\n")
    return False

  @pytest.mark.parametrize(
      "test_id, config_file, overrides",
      [
          (
              "deepseek3",
              "src/maxtext/configs/models/deepseek3-test.yml",
              {
                  "compile_topology": "v6e-4",
                  "base_num_decoder_layers": 4,
                  "per_device_batch_size": 1,
                  "max_target_length": 128,
              },
          ),
          (
              "llama3_8b",
              "src/maxtext/configs/models/llama3-8b.yml",
              {
                  "compile_topology": "v6e-4",
                  "base_num_decoder_layers": 4,
                  "per_device_batch_size": 1,
                  "max_target_length": 128,
              },
          ),
          (
              "qwen3_1.7b",
              "src/maxtext/configs/models/qwen3-1.7b.yml",
              {
                  "compile_topology": "v6e-4",
                  "base_num_decoder_layers": 4,
                  "per_device_batch_size": 1,
                  "max_target_length": 128,
              },
          ),
      ],
  )
  def test_hlo_diff(self, test_id, config_file, overrides):
    """Test HLO diff for parameterized configurations."""
    local_landing_dir = os.path.join(os.path.dirname(__file__), f"hlo_diff_dump_{test_id}")

    # Clean up before run
    if os.path.exists(local_landing_dir):
      shutil.rmtree(local_landing_dir)
    os.makedirs(local_landing_dir, exist_ok=True)

    try:
      base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
      config_path = os.path.join(base_dir, config_file)

      # Arguments for train_compile
      test_args = [
          None,
          config_path,
          "dataset_type=synthetic",
          "override_model_config=true",
          "compile_topology_num_slices=1",
      ]

      for key, value in overrides.items():
        test_args.append(f"{key}={value}")

      test_args.append(
          f'compile_xla_flags="--xla_dump_to={local_landing_dir} '
          f'--xla_dump_hlo_as_text=True --xla_dump_hlo_module_re=jit_train_step"'
      )

      print(f"Running train_compile for {test_id} with args:", test_args)

      # Initialize flags if not already done to avoid ParseCommandLine error
      if not flags.FLAGS.is_parsed():
        flags.FLAGS(["dummy_program", "--grain_train_files=dummy"], known_only=True)

      train_compile.main(tuple(test_args))

      print(f"Files in landing dir for {test_id}:", os.listdir(local_landing_dir))

      # Locate the dumped HLO file
      files = os.listdir(local_landing_dir)
      matches = [f for f in files if f.endswith(".after_optimizations.txt")]
      dumped_hlo = matches[0] if matches else None

      assert dumped_hlo, f"Dumped HLO file not found for {test_id}!"

      dumped_hlo_path = os.path.join(local_landing_dir, dumped_hlo)

      reference_hlo_path = os.path.join(os.path.dirname(__file__), f"../utils/reference_hlo_{test_id}.txt")

      if not os.path.exists(reference_hlo_path):
        print(f"Reference file not found. Creating it at {reference_hlo_path}")
        with (
            open(dumped_hlo_path, "r", encoding="utf-8") as f_in,
            open(reference_hlo_path, "w", encoding="utf-8") as f_out,
        ):
          count = 0
          for line in f_in:
            line = filter_line(line)
            if line is None:
              continue
            f_out.write(line)
            count += 1
            if count >= MAX_LINES:
              break
        print(f"Reference file created for {test_id}. Please commit it.")
      else:
        assert self.check_files_equal_ignoring_sections(dumped_hlo_path, reference_hlo_path), (
            f"HLO deviation detected in {test_id}! If this is intended, please run the 'Update HLO References' workflow "
            f"from Github Actions against your branch to update the reference HLO. "
            f"For more details, see docs/development/hlo_diff_testing.md."
        )
        print(f"HLO comparison successful against {reference_hlo_path}")

    finally:
      if os.path.exists(local_landing_dir):
        shutil.rmtree(local_landing_dir)


if __name__ == "__main__":
  unittest.main()
