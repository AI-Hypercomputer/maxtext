"""Tests for the checkpoint shape validator."""

import unittest
import os
from src.maxtext.experimental.agent.checkpoint_validation_agent.checkpoint_shape_validator import load_shapes, check_mismatches


class TestValidator(unittest.TestCase):
  """Test suite for shape validation logic."""

  def test_load_shapes_parsing(self):
    # create a temp file
    with open("test_shapes.txt", "w", encoding="utf-8") as f:
      f.write("key: layer_0 | shape: (10, 10)\n")

    shapes = load_shapes("test_shapes.txt")
    self.assertEqual(shapes["layer_0"], "(10, 10)")
    os.remove("test_shapes.txt")

  def test_logic_detects_mismatch(self):
    # pass pure dictionaries to the function to simulate a mismatch
    ideal = {"layer_0": "(10, 10)"}
    actual = {"layer_0": "(10, 11)"}  # deliberate mismatch

    # function should return True indicating a mismatch exists
    has_mismatch = check_mismatches(ideal, actual)
    self.assertTrue(has_mismatch)
