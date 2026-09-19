# Copyright 2023-2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "innovation" basis,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the checkpoint shape validator."""

import unittest
import os
import tempfile
from maxtext.experimental.agent.ckpt_validation_pipeline.checkpoint_shape_validator import load_shapes, check_mismatches


class TestValidator(unittest.TestCase):
  """Test suite for shape validation logic."""

  def test_load_shapes_parsing(self):
    with tempfile.NamedTemporaryFile(mode="w", delete=False, encoding="utf-8") as f:
      f.write("key: layer_0 | shape: (10, 10)\n")
      temp_name = f.name
    self.addCleanup(os.remove, temp_name)
    shapes = load_shapes(temp_name)
    self.assertEqual(shapes["layer_0"], "(10, 10)")

  def test_logic_detects_mismatch(self):
    # pass pure dictionaries to the function to simulate a mismatch
    ideal = {"layer_0": "(10, 10)"}
    actual = {"layer_0": "(10, 11)"}  # deliberate mismatch

    # function should return True indicating a mismatch exists
    has_mismatch, _ = check_mismatches(ideal, actual)
    self.assertTrue(has_mismatch)

  def test_logic_detects_missing_keys(self):
    ideal = {"layer_0": "(10, 10)", "layer_1": "(5, 5)"}
    actual = {"layer_0": "(10, 10)"}
    has_mismatch, mismatched_layers = check_mismatches(ideal, actual)
    self.assertTrue(has_mismatch)
    self.assertIn("layer_1", mismatched_layers)
