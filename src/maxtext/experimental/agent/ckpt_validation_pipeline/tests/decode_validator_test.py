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

"""Unit tests for the Checkpoint Validation Agent."""

import unittest
from unittest.mock import patch, MagicMock
from src.maxtext.experimental.agent.ckpt_validation_pipeline.decode_validator import (
    validate_checkpoint,
)

class TestCheckpointValidationAgent(unittest.TestCase):
  """Test suite for the checkpoint validation agent."""

  def test_missing_strict_architecture_flags(self):
    """test that the script blocks execution if scan_layers or tokenizer is missing."""
    with self.assertRaisesRegex(ValueError, "REQUIRED: You must provide 'scan_layers'"):
      # Missing scan_layers
      validate_checkpoint("test-run", "qwen", "gs://fake", "", ["tokenizer_path=fake/path"])

    with self.assertRaisesRegex(ValueError, "REQUIRED: You must provide 'tokenizer_path'"):
      # Missing tokenizer_path
      validate_checkpoint("test-run", "qwen", "gs://fake", "", ["scan_layers=false"])

  @patch("src.maxtext.experimental.agent.ckpt_validation_pipeline.decode_validator.subprocess.run")
  @patch("os.makedirs")
  @patch("builtins.open")
  def test_successful_command_generation(self, _mock_open, _mock_makedirs, mock_subprocess):
    """test that the script correctly builds the right MaxText command."""
    mock_subprocess.return_value = MagicMock(returncode=0, stderr="")

    validate_checkpoint(
        "success-test",
        "qwen3-4b",
        "gs://path/to/checkpoint",
        "",
        ["tokenizer_path=Qwen/Qwen3-4B", "scan_layers=False", "per_device_batch_size=16.0"]
    )

    mock_subprocess.assert_called_once()
    executed_command = mock_subprocess.call_args[0][0]

    self.assertIn("run_name=success-test", executed_command)
    self.assertIn("model_name=qwen3-4b", executed_command)
    self.assertIn("load_parameters_path=gs://path/to/checkpoint", executed_command)
    self.assertIn("scan_layers=False", executed_command)
    self.assertIn("per_device_batch_size=16.0", executed_command)

  @patch("src.maxtext.experimental.agent.ckpt_validation_pipeline.decode_validator.subprocess.run")
  @patch("os.makedirs")
  @patch("builtins.open")
  @patch("maxtext.utils.gcs_utils.upload_blob")
  def test_upload_to_gcs(self, mock_upload_blob, _mock_open, _mock_makedirs, mock_subprocess):
    """test that GCS upload uses the official maxtext utility."""
    mock_subprocess.return_value = MagicMock(returncode=0, stderr="")

    validate_checkpoint(
        "success-test",
        "qwen3-4b",
        "gs://path/to/checkpoint",
        "gs://my-bucket/reports",
        ["tokenizer_path=Qwen/Qwen3-4B", "scan_layers=False"]
    )

    mock_upload_blob.assert_called_once()


if __name__ == "__main__":
  unittest.main()
