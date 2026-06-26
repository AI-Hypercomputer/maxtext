import unittest
from unittest.mock import patch, mock_open, MagicMock
import json
from src.maxtext.experimental.agent.checkpoint_validation_agent.main import validate_checkpoint

class TestCheckpointValidationAgent(unittest.TestCase):
    @patch("os.path.exists", return_value=True)
    @patch("builtins.open", new_callable=mock_open, read_data='{"run_name": "test"}')
    def test_missing_required_keys(self, mock_file, mock_exists):
        """test that the script fails fast if the JSON is missing required keys."""
        with self.assertRaisesRegex(KeyError, "CRITICAL ERROR: JSON config is missing required key"):
            validate_checkpoint("fake_path.json")

    @patch("os.path.exists", return_value=True)
    @patch("builtins.open", new_callable=mock_open, read_data=json.dumps({
        "run_name": "test-run",
        "checkpoint_gcs_path": "gs://fake",
        "maxtext_model_name": "qwen",
        "maxtext_overrides": {"tokenizer_path": "fake/path"} #scan_layers is missing
    }))
    def test_missing_strict_architecture_flags(self, mock_file, mock_exists):
        """test that the script blocks execution if scan_layers or tokenizer is missing."""
        with self.assertRaisesRegex(ValueError, "REQUIRED: You must provide 'scan_layers'"):
            validate_checkpoint("fake_path.json")

    @patch("os.path.exists", return_value=True)
    @patch("src.maxtext.experimental.agent.checkpoint_validation_agent.main.subprocess.run")
    @patch("os.makedirs")
    def test_successful_command_generation(self, mock_makedirs, mock_subprocess, mock_exists):
        """test that the script correctly parses valid JSON into the right MaxText command."""
        valid_json = {
            "run_name": "success-test",
            "checkpoint_gcs_path": "gs://fake/0/items",
            "maxtext_model_name": "qwen3-4b",
            "maxtext_overrides": {
                "tokenizer_path": "Qwen/Qwen3-4B",
                "scan_layers": False,
                "per_device_batch_size": 16.0
            }
        }
        
        #mock reading the JSON file and the report writing
        with patch("builtins.open", mock_open(read_data=json.dumps(valid_json))):
            mock_subprocess.return_value = MagicMock(returncode=0, stderr="")
            
            validate_checkpoint("fake_path.json")
            
            #assert that the subprocess was called once
            mock_subprocess.assert_called_once()
            
            #extract command list that was passed to subprocess
            executed_command = mock_subprocess.call_args[0][0]
            
            #prove dynamic injection worked
            self.assertIn("run_name=success-test", executed_command)
            self.assertIn("model_name=qwen3-4b", executed_command)
            self.assertIn("scan_layers=False", executed_command)
            self.assertIn("per_device_batch_size=16.0", executed_command)

if __name__ == "__main__":
    unittest.main()