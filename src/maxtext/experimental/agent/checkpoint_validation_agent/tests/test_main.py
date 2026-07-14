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
from unittest.mock import patch, mock_open, MagicMock
import json
from src.maxtext.experimental.agent.checkpoint_validation_agent.main import (
    validate_checkpoint,
)


class TestCheckpointValidationAgent(unittest.TestCase):
  """Test suite for the checkpoint validation agent."""

  @patch("os.path.exists", return_value=True)
  @patch("builtins.open", new_callable=mock_open, read_data='{"run_name": "test"}')
  # Add underscores to the unused mock arguments
  def test_missing_required_keys(self, _mock_file, _mock_exists):
    """test that the script fails fast if the JSON is missing required keys."""
    with self.assertRaisesRegex(KeyError, "CRITICAL ERROR: JSON config is missing required key"):
      validate_checkpoint("fake_path.json")

  @patch("os.path.exists", return_value=True)
  @patch(
      "builtins.open",
      new_callable=mock_open,
      read_data=json.dumps(
          {
              "run_name": "test-run",
              "checkpoint_gcs_path": "gs://fake",
              "maxtext_model_name": "qwen",
              "maxtext_overrides": {"tokenizer_path": "fake/path"},  # scan_layers is missing
          }
      ),
  )
  def test_missing_strict_architecture_flags(self, _mock_file, _mock_exists):
    """test that the script blocks execution if scan_layers or tokenizer is missing."""
    with self.assertRaisesRegex(ValueError, "REQUIRED: You must provide 'scan_layers'"):
      validate_checkpoint("fake_path.json")

  @patch("os.path.exists", return_value=True)
  @patch("src.maxtext.experimental.agent.checkpoint_validation_agent.main.subprocess.run")
  @patch("os.makedirs")
  def test_successful_command_generation(self, _mock_makedirs, mock_subprocess, _mock_exists):
    """test that the script correctly parses valid JSON into the right MaxText command."""
    valid_json = {
        "run_name": "success-test",
        "checkpoint_gcs_path": "gs://path/to/checkpoint",
        "maxtext_model_name": "qwen3-4b",
        "maxtext_overrides": {
            "tokenizer_path": "Qwen/Qwen3-4B",
            "scan_layers": False,
            "per_device_batch_size": 16.0,
        },
    }

    # mock reading the JSON file and the report writing
    with patch("builtins.open", mock_open(read_data=json.dumps(valid_json))):
      mock_subprocess.return_value = MagicMock(returncode=0, stderr="")

      validate_checkpoint("fake_path.json")

      # assert that the subprocess was called once
      mock_subprocess.assert_called_once()

      # extract command list that was passed to subprocess
      executed_command = mock_subprocess.call_args[0][0]

      # prove dynamic injection worked
      self.assertIn("run_name=success-test", executed_command)
      self.assertIn("model_name=qwen3-4b", executed_command)
      self.assertIn("scan_layers=False", executed_command)
      self.assertIn("per_device_batch_size=16.0", executed_command)

  @patch("src.maxtext.experimental.agent.checkpoint_validation_agent.main.subprocess.run")
  def test_upload_to_gcs(self, mock_subprocess):
    """test that upload_to_gcs correctly calls gsutil."""
    from src.maxtext.experimental.agent.checkpoint_validation_agent.main import upload_to_gcs
    upload_to_gcs("/local/report.json", "gs://my-bucket/reports")
    mock_subprocess.assert_called_once()
    executed_command = mock_subprocess.call_args[0][0]
    self.assertEqual(executed_command, ["gsutil", "cp", "/local/report.json", "gs://my-bucket/reports/report.json"])

  @patch("src.maxtext.experimental.agent.checkpoint_validation_agent.main.subprocess.run")
  def test_upload_to_gcs_invalid_path(self, mock_subprocess):
    """test that upload_to_gcs rejects invalid paths without running gsutil."""
    from src.maxtext.experimental.agent.checkpoint_validation_agent.main import upload_to_gcs
    upload_to_gcs("/local/report.json", "https://invalid/path")
    mock_subprocess.assert_not_called()


if __name__ == "__main__":
  unittest.main()
